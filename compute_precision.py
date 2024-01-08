import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import json
import pandas as pd
import time
from tqdm import tqdm
import ast

def run_mistral(model, tokenizer, question, context):

    text = f"You are a Trivia QA bot. Answer the following trivia question given the context above. Answer the question with a single word if possible. If the context does not give the answer, reply with \"I don't know\".\n"
    
    n_shot_prompting = [
        {
            "context": "Paris is the capital and most populous city of France. With an official estimated population of 2,102,650 residents as of 1 January 2023 in an area of more than 105 km2 (41 sq mi), Paris is the fifth-most populated city in the European Union and the 30th most densely populated city in the world in 2022.",
            "question": "What is the capital of France?",
            "answer": "Paris"
        },
        {
            "context": "Lee Jae-myung, the leader of the South Korean opposition, is hospitalized following a stabbing attack in Busan.",
            "question": "What was the strongest earthquake in history?",
            "answer": "I don't know"
        },
        {
            "context": "Dean Lawrence Kamen (born April 5, 1951) is an American engineer, inventor, and businessman. He is known for his invention of the Segway and iBOT,[2] as well as founding the non-profit organization FIRST with Woodie Flowers. Kamen holds over 1,000 patents.",
            "question": "Who invented the segway?",
            "answer": "Dean Kamen"
        },
        {
            "context": "The peregrine falcon is the fastest bird, and the fastest member of the animal kingdom, with a diving speed of over 300 km/h (190 mph). The fastest land animal is the cheetah. Among the fastest animals in the sea is the black marlin, with uncertain and conflicting reports of recorded speeds.",
            "question": "What is the fastest animal?",
            "answer": "Cheetah"
        },
        {
            "context": "A 1-1 tie in 26 innings was played by the Brooklyn Dodgers and Boston Braves on May 1, 1920, at Braves Field in Boston, still the most innings played in a Major League Baseball (MLB) game.",
            "question": "Where is the Eiffel Tower?",
            "answer": "I don't know"
        }
    ]
    n_shot_prompting = []

    text = f"{text}\n\n" + "\n\n".join([f"Context: {p['context']}\nQuestion: {p['question']}\nAnswer: {p['answer']}" for p in n_shot_prompting])
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
    return answer, input_ids.shape[1]

def write_results(results, output_file):
    df = pd.DataFrame(results)
    print(f"Writing {output_file}")
    df.to_json(output_file, orient="records", lines=True)

model_name = sys.argv[1]
dataset = sys.argv[2]
output_file = sys.argv[3]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map = "auto").cuda()

total_correct = 0
total_correct_and_found = 0
total_correct_and_guessed = 0
total_guessed = 0
num_found = 0
results = []
with open(dataset) as f:
    all_data = []
    for line in tqdm(f):
        data = json.loads(line)
        all_data.append(data)

    total_qs = len(all_data)
    for i, data in enumerate(all_data):
        start_time = time.time()

        # print(data)
        question = data["question"]
        search_results = data["search_results"]
        answers = data['answers']
        search_found_answer = data['found_answer']
        # print(type(answers))
        # print(answers)
        answers = ast.literal_eval(answers)
        # print(answers)
        answer = answers[0]['text']
        print(f"Question {i}/{total_qs}")

        context = "\n".join(search_results)

        guess, num_tokens = run_mistral(model, tokenizer, question, context)
        is_correct = (answer.strip().lower() in guess.strip().lower())
        print(f"Context: {context}")
        print(f"Context contains answer: {search_found_answer}")
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"?: {guess}")
        
        
        print(f"Num Tokens: {num_tokens}")

        if is_correct:
            total_correct += 1
            print(f"✅")
        else:
            print(f"❌")
        print()

        if search_found_answer:
            num_found += 1
        
        if search_found_answer and is_correct:
            total_correct_and_found += 1

        if "i don't know" not in guess.lower() and is_correct:
            total_correct_and_guessed += 1
        
        if "i don't know" not in guess.lower():
            total_guessed += 1
        

        accuracy = total_correct / float(i+1) * 100.0
        precision = total_correct_and_guessed / float(num_found) * 100.0
        precision_when_guessed = total_correct_and_guessed / float(total_guessed) * 100.0
        recall = num_found / float(i+1) * 100.0
        print(f"Accuracy {total_correct}/{i+1} = {accuracy:2f}")
        print(f"Precision {total_correct_and_guessed}/{num_found} = {precision:2f}")
        print(f"Recall {num_found}/{i+1} = {recall:2f}")
        print(f"Precision when guessed {total_correct_and_guessed}/{total_guessed} = {precision_when_guessed:2f}")
        print("="*80)

        end_time = time.time()
        
        total_time = end_time - start_time
        result = {
            "idx": i,
            "question": question,
            "context": context,
            "answer": answer,
            "guess": guess,
            "search_found_answer": search_found_answer,
            "is_correct": is_correct,
            "time": total_time,
            "num_tokens": num_tokens,
            "tokens_per_sec": (num_tokens/total_time)
        }
        results.append(result)

        if len(results) % 20 == 0:
            write_results(results, output_file)
            
        # if len(results) > 10:
        #     break
        sys.stdout.flush()

write_results(results, output_file)