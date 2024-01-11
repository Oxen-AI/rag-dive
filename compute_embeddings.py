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
import pandas as pd
import time
import os
import json
import torch
from tqdm import tqdm
import argparse

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
model = AutoModel.from_pretrained("thenlper/gte-large").cuda()

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def compute_embedding(batch, max_seq_len=512):
    input_texts = [chunk['context'] for chunk in batch]

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=max_seq_len, padding=True, truncation=True, return_tensors='pt')
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
    outfile,
    acc_chunks,
    embeddings,
):
    # Write the embeddings to a parquet file
    parent_dir = os.path.dirname(outfile)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    print(f"Saving embeddings to {outfile}")
    table = pa.Table.from_arrays(
        [
            pa.array([chunk['context'] for chunk in acc_chunks]),
            pa.array([chunk['question_ids'] for chunk in acc_chunks]),
            pa.array(embeddings),
        ],
        names=["context", "question_ids", "embedding"],
    )
    pq.write_table(table, outfile)

def embed_dataset(
    input_file: str,
    output_file: str,
    batch_size: int,
    max_seq_len: int,
):
    filename = input_file

    print("Reading file...")
    df = pd.read_parquet(filename)
    print(df)

    acc_chunks = []
    embeddings = []
    current_batch = []
    for i, batch in tqdm(df.iterrows()):
        current_batch.append(batch)
        if len(current_batch) == batch_size:
            batch_chunks, batch_embeddings = compute_embedding(current_batch)
            acc_chunks.extend(batch_chunks)
            embeddings.extend(batch_embeddings)
            current_batch = []

    if len(current_batch) > 0:
        batch_chunks, batch_embeddings = compute_embedding(current_batch)
        acc_chunks.extend(batch_chunks)
        embeddings.extend(batch_embeddings)

    save_embeddings(
        output_file,
        acc_chunks,
        embeddings
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input file to read from.", required=True)
    parser.add_argument("-o", "--output", type=str, help="Output dataset to write to.", required=True)
    parser.add_argument("-b", "--batch_size", type=int, help="Size of the batch to compute", default=16)
    parser.add_argument("-m", "--max_seq_len", type=int, help="Max size of the sequence", default=512)
    args = parser.parse_args()

    embed_dataset(
        input_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len
    )
