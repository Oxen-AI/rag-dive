
import chromadb
import pandas as pd
import time
from tqdm import tqdm
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def compute_embedding(text):
    input_texts = [
        text
    ]

    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
    model = AutoModel.from_pretrained("thenlper/gte-large")

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    # print(embeddings)

    # convert embeddings to list of lists
    embeddings = embeddings.tolist()
    # print(embeddings)
    return embeddings

# Connect to chromadb
chroma_client = chromadb.PersistentClient(path="chroma.db")

collection_name = "squad_embeddings"
print(f"Reading collection {collection_name}...")

collection = chroma_client.get_collection(name=collection_name)

while True:
    text = input("> ")
    embeddings = compute_embedding(text)
    result = collection.query(
        query_embeddings=embeddings,
        n_results=10,
    )
    
    print("Results:")
    print(result)
    print("*"*80)

    metadatas = result['metadatas'][0]
    documents = result['documents'][0]

    for i, document in enumerate(documents):
        question_ids = metadatas[i]['question_ids'].split(",")
        print(question_ids)
        print(document)
        print()
