
import chromadb
import pandas as pd
import time
import json
from tqdm import tqdm
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import sys
import argparse

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
model = AutoModel.from_pretrained("thenlper/gte-large")# .to('cuda')

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def compute_embedding(text):
    input_texts = [
        text
    ]

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')# .to('cuda')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    # print(embeddings)

    # convert embeddings to list of lists
    embeddings = embeddings.tolist()
    # print(embeddings)
    return embeddings

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, required=True)
    parser.add_argument("-d", "--database", type=str, required=True)
    parser.add_argument("-o", "--output_file", type=str, required=True)
    parser.add_argument("-c", "--collection", type=str, default="squad_embeddings")
    parser.add_argument("-n", "--top_n", type=int, default=10, help="Number of results to return per query.")
    parser.add_argument("-l", "--limit", type=int, help="Number of examples to run for. If -1 will run for all.", default=-1)
    parser.add_argument("-f", "--filter", type=str, help="Answers we want to filter out.", default=None)
    args = parser.parse_args()
    
    examples_file = args.input_file # "SQuAD/val.jsonl"
    chroma_db = args.database # "chroma.db"
    results_file = args.output_file # "SQuAD/results.jsonl"

    # Connect to chromadb
    chroma_client = chromadb.PersistentClient(path=chroma_db)

    collection_name = args.collection
    print(f"Reading collection {collection_name}...")

    collection = chroma_client.get_collection(name=collection_name)

    examples = []
    with open(examples_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line))

    results = []
    num_results_per_query = args.top_n # 10
    num_found = 0
    for i, example in enumerate(examples):
        if args.limit > 0 and len(results) >= args.limit:
            break
        
        if args.filter != None and example['answer'] == args.filter:
            continue

        print("="*40 + f"START {i}" + "="*40)
        start_time = time.time()
        start_embedding_time = time.time()
        question = example['question']
        q_id = example['id']
        answer = example['answer']
        print(f"Querying for {q_id} -> {question}")
        embeddings = compute_embedding(question)
        end_embedding_time = time.time()
        
        start_search_time = time.time()
        result = collection.query(
            query_embeddings=embeddings,
            n_results=num_results_per_query,
        )
        end_search_time = time.time()
        
        print(f"Question: {question}")
        print(f"Answer: {answer}")

        metadatas = result['metadatas'][0]
        documents = result['documents'][0]

        found_it = False
        found_idx = -1
        for j, document in enumerate(documents):
            question_ids = metadatas[j]['question_ids'].split(",")
            # print(question_ids)
            # print(document)
            # print()
            
            if q_id in question_ids:
                print(f"✅ Found @{j}")
                print(f"Context: {document}")
                print()
                num_found += 1
                found_it = True
                found_idx = j
                break

        if not found_it:
            print(f"❌ Could not find context for {q_id}")
            print()

        end_time = time.time()
        total_time = end_time - start_time
        results.append({
            "question_id": q_id,
            "question": question,
            "answer": answer,
            "search_results": documents,
            "found_idx": found_idx,
            "n_results": num_results_per_query,
            "embedding_time": end_embedding_time - start_embedding_time,
            "search_time": end_search_time - start_search_time,
            "total_time": total_time,
            "found_answer": found_it
        })

        total = len(results)
        percentage = num_found / (total+1)
        print(f"Total time: {total_time}")
        print(f"Found {num_found}/{total+1} = {percentage} questions")
        print("="*40 + f"END {i}" + "="*40)

        if i % 10 == 0:
            df = pd.DataFrame(results)
            df.to_json(results_file, orient="records", lines=True)
            print("Saved to file")

    df = pd.DataFrame(results)
    df.to_json(results_file, orient="records", lines=True)
    
if __name__ == "__main__":
    main()