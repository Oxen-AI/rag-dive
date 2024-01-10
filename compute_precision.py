import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import json
import pandas as pd
import time
from tqdm import tqdm
import argparse

def construct_prompt(question, context, n=0):
    text = f"You are a Trivia QA bot. Answer the following trivia question given the context above. Answer the question with a single word if possible. If the context does not give the answer, reply with \"not_in_context\".\n"

    n_shot_prompting = [
        {
            "context": "Paris is the capital and most populous city of France. With an official estimated population of 2,102,650 residents as of 1 January 2023 in an area of more than 105 km2 (41 sq mi), Paris is the fifth-most populated city in the European Union and the 30th most densely populated city in the world in 2022.",
            "question": "What is the capital of France?",
            "answer": "Paris"
        },
        {
            "context": "Lee Jae-myung, the leader of the South Korean opposition, is hospitalized following a stabbing attack in Busan.",
            "question": "What was the strongest earthquake in history?",
            "answer": "not_in_context"
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
            "answer": "not_in_context"
        }
    ]
    if n > 0:
        n_shot_prompting = n_shot_prompting[:n]
    else:
        n_shot_prompting = []

    text = f"{text}\n\n" + "\n\n".join([f"Context: {p['context']}\nQuestion: {p['question']}\nAnswer: {p['answer']}" for p in n_shot_prompting])
    text = f"{text}\n\nContext: {context}\nQuestion: {question}\nAnswer: "
    return text

def run_model(model, tokenizer, question, context, n_shot=0, end_instruct="[/INST]"):
    prompt = construct_prompt(question, context, n=n_shot)

    # print(text)

    messages = [
        {"role": "user", "content": prompt}
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
    print("="*80)
    print(decoded)
    print("="*80)

    # out returns the whole sequence plus the original
    cleaned = decoded.split(end_instruct)[-1]
    cleaned = cleaned.replace("</s>", "")

    # # the model will just keep generating, so only grab the first one
    answer = cleaned.split("\n\n")[0].strip()
    # answer = cleaned.strip()
    # print(answer)
    return answer

def run_together_ai(model, question, context, n_shot=0):
    import openai
    import os

    client = openai.OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY"),
        base_url='https://api.together.xyz',
    )

    prompt = construct_prompt(question, context, n=n_shot)

    # print(text)

    messages = [
        {"role": "user", "content": prompt}
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model, # "mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_tokens=1024
    )

    answer = chat_completion.choices[0].message.content
    return answer

def run_openai(model, question, context, n_shot=0):
    import openai
    import os

    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    prompt = construct_prompt(question, context, n=n_shot)

    # print(text)

    messages = [
        {"role": "user", "content": prompt}
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        max_tokens=1024
    )

    answer = chat_completion.choices[0].message.content
    return answer

def run_llama_cpp(model, question, context, n_shot=0):
    from llama_cpp import Llama

    prompt = construct_prompt(question, context, n=n_shot)

    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    llm = Llama(
        model_path=model, # "./mistral-7b-instruct-v0.2.Q4_K_M.gguf",  # Download the model file first
        n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
        n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
        n_gpu_layers=35         # The number of layers to offload to GPU, if you have GPU acceleration available
    )

    # Simple inference example
    # chat_completion = llm(
    #     f"<s>[INST] {prompt} [/INST]", # Prompt
    #     max_tokens=512,  # Generate up to 512 tokens
    #     stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
    #     echo=True        # Whether to echo the prompt
    # )

    chat_completion = llm.create_chat_completion(
        messages = [
            {"role": "user", "content": prompt}
        ]
    )
    print(chat_completion)

    answer = chat_completion['choices'][0]['message']['content'].strip()
    return answer

def model_guessed_not_in_context(guess):
    return "not_in_context" in guess.lower() or "not in context" in guess.lower()

def answer_is_correct(answer, guess):
    return answer.strip().lower() in guess.strip().lower() or guess.strip().lower() in answer.strip().lower()

def write_results(results, output_file):
    df = pd.DataFrame(results)
    print(f"Writing {output_file}")
    df.to_json(output_file, orient="records", lines=True)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="HuggingFace model name.")
parser.add_argument("-s", "--service", type=str, help="Service to use, options: hugging_face, openai, together_ai.")
parser.add_argument("-d", "--dataset", type=str, help="Dataset to run on.", required=True)
parser.add_argument("-o", "--output_file", type=str, help="Output file to write results to.", required=True)
parser.add_argument("-e", "--end_instruct", type=str, default="[/INST]", help="String to end the instruction prompt.")
parser.add_argument("-l", "--context_length", type=int, default=1, help="How many documents we have in the context window")
parser.add_argument("-n", "--n_shot", type=int, default=0, help="Number of examples to give for N-Shot Prompt")
args = parser.parse_args()

model_name = args.model_name
dataset = args.dataset
output_file = args.output_file

if args.service == "openai" or args.service == "together_ai" or args.service == "llama_cpp":
    model = args.model_name
else:
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
        answer = data['answer']
        search_found_answer = data['found_answer']
        found_idx = data['found_idx']
        search_found_answer = search_found_answer and found_idx < args.context_length
        # print(type(answers))
        # print(answers)
        # print(answers)
        print(f"Question {i}/{total_qs}")

        context = "\n".join(search_results[:args.context_length])

        if args.service == "openai":
            guess = run_openai(model, question, context, n_shot=args.n_shot)
        elif args.service == "together_ai":
            guess = run_together_ai(model, question, context, n_shot=args.n_shot)
        elif args.service == "llama_cpp":
            guess = run_llama_cpp(model, question, context, n_shot=args.n_shot)
        else:
            guess = run_model(model, tokenizer, question, context, n_shot=args.n_shot, end_instruct=args.end_instruct)

        is_correct = answer_is_correct(answer, guess)
        print(f"Context: {context}")
        print(f"Context contains answer: {search_found_answer}")
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"?: {guess}")

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

        if not model_guessed_not_in_context(guess) and is_correct:
            total_correct_and_guessed += 1

        if not model_guessed_not_in_context(guess):
            total_guessed += 1

        accuracy = total_correct / float(i+1) * 100.0
        if num_found > 0:
            precision = total_correct_and_guessed / float(num_found) * 100.0
        else:
            precision = 0.0
        if total_guessed > 0:
            precision_when_guessed = total_correct_and_guessed / float(total_guessed) * 100.0
        else:
            precision_when_guessed = 0.0
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
            "question_id": data["question_id"],
            "question": question,
            "context": context,
            "answer": answer,
            "guess": guess,
            "search_found_answer": search_found_answer,
            "is_correct": is_correct,
            "time": total_time
        }
        results.append(result)

        if len(results) % 20 == 0:
            write_results(results, output_file)

        # if len(results) > 10:
        #     break
        sys.stdout.flush()

write_results(results, output_file)
