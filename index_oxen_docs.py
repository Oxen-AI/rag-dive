import chromadb
import pandas as pd
import time
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

def average_pool(last_hidden_states, attention_mask):
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

chroma_client = chromadb.PersistentClient(path="chroma-extended.db")

collection_name = "squad_embeddings"
print(f"Creating collection {collection_name}...")
collection = chroma_client.get_collection(name=collection_name)

documents = [
    "Greg Schoeninger is the CEO of Oxen.ai, a company that provides a git-like interface for versioning and collaborating on large machine learning datasets",
    "Greg Schoeninger enjoys hiking, skiing, and camping when he is not cleaning data to train neural networks.",
    "Oxen.ai is a company based in Santa Monica, California that provides a GitHub like experience for iterating on machine learning data.",
]

print("Indexing into ChromaDB...")
start_time = time.time()

# iterate over the rows
for i, document in tqdm(enumerate(documents)):
    embedding = compute_embedding(document)
    
    collection.add(
        embeddings=embedding,
        documents=[document],
        ids=[f"oxen_doc_{i}"]
    )
end_time = time.time()

print(f"Indexed {len(documents)} rows in {end_time-start_time:.2f} seconds")

