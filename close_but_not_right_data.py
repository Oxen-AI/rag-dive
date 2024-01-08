
import chromadb
import pandas as pd
import time
import json
from tqdm import tqdm
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import sys
import uuid
import random
import ast

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
model = AutoModel.from_pretrained("thenlper/gte-large")

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def compute_embedding(text):
    input_texts = [
        text
    ]

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    # print(embeddings)

    # convert embeddings to list of lists
    embeddings = embeddings.tolist()
    # print(embeddings)
    return embeddings

def main():
    if len(sys.argv) != 3:
        print("Usage: python compute_recall.py <input.jsonl> <output.jsonl>")
        exit()

    # Connect to chromadb
    chroma_client = chromadb.PersistentClient(path="chroma.db")

    collection_name = "squad_embeddings"
    print(f"Reading collection {collection_name}...")

    collection = chroma_client.get_collection(name=collection_name)

    examples_file = sys.argv[1] # "SQuAD/random.jsonl"
    results_file = sys.argv[2] # "SQuAD/results.jsonl"

    examples = []
    with open(examples_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
        random.shuffle(examples)

    results = []
    num_results_per_query = 10
    num_found = 0
    for i, example in enumerate(examples):
        print("="*40 + f"START {i}" + "="*40)
        start_time = time.time()
        start_embedding_time = time.time()
        question = example['question']
        q_id = example['id']
        answers = example['answers']
        answers = ast.literal_eval(answers)
        
        if len(answers) == 0:
            continue
        
        print("Question", question)
        print("Answers", answers)
        answer = answers[0]['text']
        
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

        bad_passage = ""
        for j, document in enumerate(documents):
            question_ids = metadatas[j]['question_ids'].split(",")

            if q_id not in question_ids:
                bad_passage = document
                break

        end_time = time.time()
        results.append({
            "id": f"search-{str(uuid.uuid4())}",
            "context": bad_passage,
            "question": question,
            "answer": "answer_not_found",
            "category": "search_questions",
            "annotator": "thenlper/gte-large",
        })

        print(f"Bad Context: {bad_passage}")
        print(f"Question: {question}")
        print(f"Answer: {answers}")
        if i % 10 == 0:
            df = pd.DataFrame(results)
            df.to_json(results_file, orient="records", lines=True)
            print("Saved to file")

    df = pd.DataFrame(results)
    df.to_json(results_file, orient="records", lines=True)
    
if __name__ == "__main__":
    main()