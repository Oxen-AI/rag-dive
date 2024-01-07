
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


stub = Stub("stream_embeddings")

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
        def _substr(s, n):
            return s[:n] + "..." if len(s) > n else s

        texts = [_substr(chunk['text'], 1024) for chunk in batch]
        res = await self.client.post("/embed", json={"inputs": texts})
        return np.array(res.json())

    @method()
    async def embed(self, batch):
        """Embeds a list of texts."""

        result = [self._embed(batch)]
        embeddings = np.concatenate(await asyncio.gather(*result))
        return batch, embeddings

@stub.function(
    image=Image.debian_slim()
        .pip_install("oxenai", "pyarrow", "tqdm", "requests", "sentence_splitter"),
    volumes={cache_dir: volume},
    timeout=84600,
    secret=Secret.from_name("oxenai-api-key"),
)
def embed_dataset(
    input_dataset: str,
    input_directory: str,
    output_dataset: str,
):
    from oxen.streaming_dataset import load_dataset
    from oxen.auth import config_auth as config_oxen_auth
    from oxen.remote_repo import create_repo
    from oxen.providers.oxen_data_frame_provider import OxenDataFrameProvider
    from oxen.streaming_dataset import StreamingDataset
    from sentence_splitter import SentenceSplitter, split_text_into_sentences

    import oxen
    import pyarrow as pa
    import pyarrow.parquet as pq
    import time
    import os
    import json
    from tqdm import tqdm
    
    splitter = SentenceSplitter(language='en')
    def generate_sentence_windows(dataset, n=2, batch_size=512, max_examples=-1):
        for i, x in enumerate(dataset):
            sentences = splitter.split(x['text'])
            sentence_window = []
            batch = []
            for s in sentences:
                sentence_window.append(s)
                if len(sentence_window) == n:
                    window = " ".join(sentence_window)
                    new_x = x.copy()
                    new_x['text'] = window
                    batch.append(new_x)
                    sentence_window = []

                if len(batch) == batch_size:
                    yield batch
                    batch = []

            if len(batch) > 0:
                yield batch

            if max_examples > 0 and i > max_examples:
                break
            

    oxenai_token = os.environ["OXENAI_API_KEY"]
    config_oxen_auth(oxenai_token)

    # Load the dataset from https://www.oxen.ai/{input_dataset}
    input_repo = oxen.RemoteRepo(input_dataset)
    paths = input_repo.ls(input_directory)

    local_repo = oxen.init(data_dir)
    remote_repo = create_repo(output_dataset)
    local_repo.set_remote("origin", remote_repo.url())

    for path in paths:
        # Stream one data file at a time from Oxen.ai
        filename = os.path.join(input_directory, path.filename)
        provider = OxenDataFrameProvider(input_repo, paths=[filename])
        dataset = StreamingDataset(provider)

        print(f"Dataset size {len(dataset)} rows")

        # Interface to compute embeddings
        model = TextEmbeddingsInference()
        sentences = generate_sentence_windows(dataset, n=2)

        acc_chunks = []
        embeddings = []
        for batch_chunks, batch_embeddings in model.embed.map(sentences, order_outputs=False):
            print(batch_chunks)
            print(batch_embeddings)
            acc_chunks.extend(batch_chunks)
            embeddings.extend(batch_embeddings)

        # Save embeddings to Oxen.ai
        basename = os.path.basename(filename)
        outfile_name = input_directory + "_" + basename.split(".")[0] + "_embeddings.parquet"

        # Write the embeddings to a parquet file
        outfile = os.path.join(data_dir, outfile_name)
        print(f"Saving embeddings to {outfile}")
        table = pa.Table.from_arrays(
            [
                pa.array([chunk['text'] for chunk in acc_chunks]),
                pa.array([chunk['source'] for chunk in acc_chunks]),
                pa.array(embeddings),
            ],
            names=["text", "source", "embedding"],
        )
        pq.write_table(table, outfile)
        
        # Add and commit files to RemoteRepo
        print(f"Adding file to repo {outfile}")
        local_repo.add(outfile)
        local_repo.commit(f"Adding embeddings for {filename}")
        local_repo.push()
        print(f"Done with {filename}")

@stub.local_entrypoint()
def main(input_dataset: str, input_directory: str, output_dataset: str):
    embed_dataset.remote(
        input_dataset=input_dataset,
        input_directory=input_directory,
        output_dataset=output_dataset
    )
