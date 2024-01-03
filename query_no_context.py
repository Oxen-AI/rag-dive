import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import sys
import json
import pandas as pd
import time
from tqdm import tqdm
import ast
import chromadb

def generate_response(model, tokenizer, question):

    text = f"You are a Trivia QA bot. Answer the following trivia question. Answer the question with a single word if possible. If you do not know the answer, reply with \"I don't know\".\n"
    
    text = f"{text}\n\nQuestion: {question}\nAnswer: "
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

model_name = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map = "auto").cuda()

num_docs = 3
print("\n🤖 Ask me anything!")
while True:
    question = input('> ')
    print("Generating response...")
    guess = generate_response(model, tokenizer, question)
    print("\n\nAnswer:")
    print(guess)
    print("\n")
    print("🤖 What else do you want to know?")
