import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import sys
import json
import pandas as pd
import time
from tqdm import tqdm
import ast
import chromadb

def generate_response(model, tokenizer, question, context):

    text = f"You are a Trivia QA bot. Answer the following trivia question given the context above. Answer the question with a single word if possible. If the context does not give the answer, reply with \"I don't know\".\n"
    
    text = f"{text}\n\nContext: {context}\nQuestion: {question}\nAnswer: "
    # print(text)
    
    messages = [
        {"role": "user", "content": text}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    
    # print(text)
    # input_ids = torch.LongTensor([tokenizer.encode(text)]).cuda()
    input_ids = encodeds.cuda()
    # print(input_ids)
    
    out = model.generate(
        input_ids,
        temperature=0.9,
        max_new_tokens=128,
        do_sample=True
    )

    # print(out)
    decoded = tokenizer.batch_decode(out)[0]
    # print("="*80)
    # print(decoded)
    # print("="*80)

    # out returns the whole sequence plus the original
    cleaned = decoded.split("[/INST]")[-1]
    cleaned = cleaned.replace("</s>", "")
    
    # # the model will just keep generating, so only grab the first one
    answer = cleaned.split("\n\n")[0].strip()
    # answer = cleaned.strip()
    # print(answer)
    return answer

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

def retrieve_documents(embeddings, n=3):
    result = collection.query(
        query_embeddings=embeddings,
        n_results=n,
    )
    return result['documents'][0]

model_name = sys.argv[1]
database_name = sys.argv[2]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map = "auto").cuda()

# Connect to chromadb
chroma_client = chromadb.PersistentClient(path=database_name)

collection_name = "squad_embeddings"
print(f"Reading collection {collection_name}...")

collection = chroma_client.get_collection(name=collection_name)

num_docs = 3
print("\n🤖 Ask me anything!")
while True:
    question = input('> ')
    print("Computing embedding...")
    embedding = compute_embedding(question)
    print(f"Retrieving {num_docs} documents...\n")
    documents = retrieve_documents(embedding, n=num_docs)
    context = "\n\n".join(documents)
    print("*"*40 + "Context" + "*"*40)
    print(context)
    print("*"*40 + "End Context" + "*"*40)
    print("Reading context and generating response...")
    guess = generate_response(model, tokenizer, question, context)
    print("\n\nAnswer:")
    print(guess)
    print("\n")
    print("🤖 What else do you want to know?")
