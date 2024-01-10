import pandas as pd
import time
from tqdm import tqdm
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import argparse
import uuid

# parse arguments for input_file, output_db, should_create_collection
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", type=str)
parser.add_argument("-o", "--output_db", type=str)
parser.add_argument("-c", "--collection", type=str, default="squad_embeddings")
parser.add_argument("-a", "--append", action="store_true")
args = parser.parse_args()

input_file = args.input_file
output_db = args.output_db
collection_name = args.collection
should_create = not args.append

chroma_client = chromadb.PersistentClient(path=output_db)

print(f"Creating collection {collection_name}...")
if should_create:
    collection = chroma_client.create_collection(name=collection_name)
else:
    collection = chroma_client.get_collection(name=collection_name)

embeddings_file = input_file
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
        ids=[str(uuid.uuid4())]
    )
end_time = time.time()

print(f"Indexed {len(df)} rows in {end_time-start_time:.2f} seconds")
