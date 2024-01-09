from oxen.streaming_dataset import load_dataset
from oxen.auth import config_auth as config_oxen_auth
from oxen.remote_repo import create_repo
from oxen.providers.oxen_data_frame_provider import OxenDataFrameProvider
from oxen.streaming_dataset import StreamingDataset
from sentence_splitter import SentenceSplitter, split_text_into_sentences
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

import oxen
import pyarrow as pa
import pyarrow.parquet as pq
import time
import os
import json
import torch
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
model = AutoModel.from_pretrained("thenlper/gte-base").cuda()

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def compute_embedding(batch):
    def _substr(s, n):
            return s[:n] + "..." if len(s) > n else s

    input_texts = [_substr(chunk['text'], 512) for chunk in batch]
    # input_texts = [item['text'] for item in batch]

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    batch_dict = {k: torch.tensor(v).to("cuda") for k, v in batch_dict.items()}
    # print(batch_dict)

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    # print(embeddings)

    # convert embeddings to list of lists
    embeddings = embeddings.tolist()
    # print(embeddings)
    return batch, embeddings

def save_embeddings(
    local_repo,
    data_dir,
    filename,
    acc_chunks,
    embeddings
):
    # Save embeddings to Oxen.ai
    basename = os.path.basename(filename)
    outfile_name = basename.split(".")[0] + f"_embeddings.parquet"

    # Write the embeddings to a parquet file
    outfile = os.path.join(data_dir, outfile_name)
    parent_dir = os.path.dirname(outfile)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    
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

def embed_dataset(
    input_dataset: str,
    input_file: str,
    output_dataset: str,
):
    oxenai_token = os.environ["OXENAI_API_KEY"]
    config_oxen_auth(oxenai_token)

    # Load the dataset from https://www.oxen.ai/{input_dataset}
    input_repo = oxen.RemoteRepo(input_dataset)

    data_dir = output_dataset.split("/")[-1]
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    local_repo = oxen.init(data_dir)
    remote_repo = create_repo(output_dataset)
    local_repo.set_remote("origin", remote_repo.url())
    # local_repo.set_remote("origin", "https://hub.oxen.ai/oxbot/PastureEmbeds")

    # Stream one data file at a time from Oxen.ai
    filename = input_file
    provider = OxenDataFrameProvider(input_repo, paths=[filename])
    dataset = StreamingDataset(provider, num_buffers=100)

    print(f"Dataset size {len(dataset)} rows")

    acc_chunks = []
    embeddings = []
    for i, batch in enumerate(tqdm(dataset)):
        batch_chunks, batch_embeddings = compute_embedding(batch)
        # print(batch_chunks)
        # print(batch_embeddings)
        acc_chunks.extend(batch_chunks)
        embeddings.extend(batch_embeddings)


    save_embeddings(
        local_repo,
        data_dir,
        filename,
        acc_chunks,
        embeddings,
    )

if __name__ == '__main__':
    embed_dataset(
        input_dataset="datasets/Not-In-Context",
        input_file="dev_contexts.jsonl",
        output_dataset="oxbot/SQuAD-Dev-Embed-2"
    )
