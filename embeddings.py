
import os
import asyncio
import subprocess

from pathlib import Path
from modal import Image, Stub, Volume, gpu, method, Secret


from util import generate_batches

GPU_CONFIG = gpu.A10G()
MODEL_ID = "thenlper/gte-large"
BATCH_SIZE = 512
PORT_NUM = 8000
N_GPU = 10

volume = Volume.persisted("embedding-wikipedia")
cache_dir = "/data"
data_dir = f"{cache_dir}/embeddings"
DATA_PATH = Path(data_dir)
# set OXEN_CONFIG_DIR environment var to data_dir
os.environ["OXEN_CONFIG_DIR"] = data_dir

dataset_name = "oxbot/SQuAD-Ctx-Embeddings"
dataset_file = "embeddings.parquet"


# https://huggingface.co/docs/text-embeddings-inference/index
DOCKER_IMAGE = (
    "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0"  # Ampere 86 for A10s.
)

LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    str(PORT_NUM),
    "--max-client-batch-size",
    str(BATCH_SIZE),
    "--max-batch-tokens",
    str(BATCH_SIZE*BATCH_SIZE),
]


def spawn_server() -> subprocess.Popen:
    import socket

    process = subprocess.Popen(["text-embeddings-router"] + LAUNCH_FLAGS)

    # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
    while True:
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            print("Webserver ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webserving process has exited.
            # If so, a connection can never be made.
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(f"launcher exited unexpectedly with code {retcode}")


def download_model():
    # Wait for server to start. This downloads the model weights when not present.
    spawn_server()


stub = Stub("embeddings")

tei_image = (
    Image.from_registry(
        "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0",
        add_python="3.10",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(download_model, gpu=GPU_CONFIG)
    .pip_install("httpx")
)


with tei_image.imports():
    import numpy as np


@stub.cls(
    gpu=GPU_CONFIG,
    image=tei_image,
    # Use up to 10 GPU containers at once.
    concurrency_limit=N_GPU,
    retries=3,
)
class TextEmbeddingsInference:
    def __enter__(self):
        # If the process is running for a long time, 
        # the client does not seem to close the connections, results in a pool timeout
        from httpx import AsyncClient

        self.process = spawn_server()
        self.client = AsyncClient(base_url="http://127.0.0.1:8000", timeout=30)

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.process.terminate()

    async def _embed(self, batch):
        # print(f"Processing text {chunk_batch[:80]}")
        
        def _substr(s, n):
            return s[:n] + "..." if len(s) > n else s
        
        texts = [_substr(chunk['context'], 1024) for chunk in batch]
        res = await self.client.post("/embed", json={"inputs": texts})
        return np.array(res.json())

    @method()
    async def embed(self, batch):
        """Embeds a list of texts.  context, question_ids = chunks[0]"""

        result = [self._embed(batch)]
        embeddings = np.concatenate(await asyncio.gather(*result))
        return batch, embeddings

@stub.function(
    image=Image.debian_slim()
        .pip_install("oxenai", "pyarrow", "tqdm", "requests"),
    volumes={cache_dir: volume},
    timeout=84600,
    secret=Secret.from_name("oxenai-api-key"),
)
def embed_dataset(
    sample_size: int = 50,
    batch_size: int = 512 * 50
):
    from oxen.streaming_dataset import load_dataset
    from oxen.auth import config_auth as config_oxen_auth
    from oxen.remote_repo import create_repo

    import oxen
    import pyarrow as pa
    import pyarrow.parquet as pq
    import time
    import os
    import json
    from tqdm import tqdm

    start = time.perf_counter()
    # Load the dataset from https://www.oxen.ai/datasets/Wikipedia
    print("Streaming dataset from Oxen.ai...")
    # dataset = load_dataset("ox/SQuAD-Context", paths="contexts.jsonl")
    repo_path = os.path.join(data_dir, "SQuAD-Context")
    repo = oxen.clone("ox/SQuAD-Context", path=repo_path)
    
    input_file = os.path.join(repo_path, "contexts.jsonl")
    
    # Load the dataset from the local file
    dataset = []
    with open(input_file, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))

    
    print(f"Dataset loaded in {time.perf_counter()-start:.2f} seconds")
    print(f"Dataset size {len(dataset)} rows")

    # Generate a subset of the dataset
    subset = []
    for i in tqdm(range(sample_size)):
        subset.append(dataset[i])

    # Interface to compute embeddings
    model = TextEmbeddingsInference()

    batches = generate_batches(subset, batch_size=batch_size)

    acc_chunks = []
    embeddings = []
    for batch_chunks, batch_embeddings in model.embed.map(batches, order_outputs=False):
        acc_chunks.extend(batch_chunks)
        embeddings.extend(batch_embeddings)

    # Save embeddings to Oxen.ai
    print(f"Pushing to hub {dataset_name}")
    oxenai_token = os.environ["OXENAI_API_KEY"]
    config_oxen_auth(oxenai_token)

    # Initialize the Oxen Repository
    repo = oxen.init(data_dir)

    # Write the embeddings to a parquet file
    table = pa.Table.from_arrays(
        [
            pa.array([chunk['context'] for chunk in acc_chunks]),
            pa.array([chunk['question_ids'] for chunk in acc_chunks]),
            pa.array(embeddings),
        ],
        names=["context", "question_ids", "embedding"],
    )
    pq.write_table(table, os.path.join(data_dir, dataset_file))
    repo.add(dataset_file)
    print(repo.status())
    repo.commit("Adding embeddings")

    remote_repo = create_repo(dataset_name)
    print("Created remote repo")
    print(remote_repo)
    print(remote_repo.url())

    repo.set_remote("origin", remote_repo.url())
    repo.push()
    


@stub.local_entrypoint()
def main():
    embed_dataset.remote(sample_size=19_028, batch_size=10)
