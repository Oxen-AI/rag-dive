import chromadb
import pandas as pd
import time
from tqdm import tqdm

chroma_client = chromadb.PersistentClient(path="chroma.db")

collection_name = "squad_embeddings"
print(f"Creating collection {collection_name}...")
collection = chroma_client.create_collection(name=collection_name)

embeddings_file = "embeddings.parquet"
print(f"Reading embeddings file {embeddings_file}...")
df = pd.read_parquet(embeddings_file)

print(df.head())

print("Indexing into ChromaDB...")
start_time = time.time()

# iterate over the rows
for index, row in tqdm(df.iterrows()):
    question_ids_str = ",".join([str(x) for x in row["question_ids"]])
    collection.add(
        embeddings=[row["embedding"].tolist()],
        documents=[row["context"]],
        metadatas=[{"question_ids": question_ids_str}],
        ids=[f"{index}"]
    )
end_time = time.time()

print(f"Indexed {len(df)} rows in {end_time-start_time:.2f} seconds")

