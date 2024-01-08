system_prompt = """
You are a Question Generator. Generate an erroneous question that does not make sense given the context. Make the questions as realistic as possible, and contain some link to the context, just like a human would ask. The questions should be as diverse as possible, and not be repetitive. The questions should not be able to be answered by the context.

The following are a few examples, extrapolate new questions given these examples.

Context: Beyoncé Giselle Knowles was born in Houston, Texas, to Celestine Ann "Tina" Knowles (née Beyincé), a hairdresser and salon owner, and Mathew Knowles, a Xerox sales manager.
Question: What is Beyoncé's mother's name?

Context: Chopin's health continued to deteriorate, particularly from this time onwards. Modern research suggests that apart from any other illnesses, he may also have suffered from temporal lobe epilepsy.
Question: What is Chopin's favorite color?

Context: Tsai writes that shortly after the visit by Deshin Shekpa, the Yongle Emperor ordered the construction of a road and of trading posts in the upper reaches of the Yangzi and Mekong Rivers in order to facilitate trade with Tibet in tea, horses, and salt.
Question: How many children does the Yongle Emperor have?

Context: Paris is the capital and most populous city of France. With an official estimated population of 2,102,650 residents as of 1 January 2023[2] in an area of more than 105 km2 (41 sq mi),[5] Paris is the fourth-most populated city in the European Union and the 30th most densely populated city in the world in 2022.
Question: What is the capital of Paris?
"""

import openai
import os, sys
import json
import uuid
import random

# require two arguments
if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <contexts_file>")
    sys.exit(1)

def write_questions_to_jsonl(questions):
    with open("SQuAD/synthetic_questions.jsonl", "w") as f:
        for question in questions:
            f.write(json.dumps(question) + "\n")

with open(sys.argv[1], "r") as f:
    contexts = [json.loads(line) for line in f]
    # shuffle the contexts
    random.shuffle(contexts)

client = openai.OpenAI(
  api_key=os.environ.get("TOGETHER_API_KEY"),
  base_url='https://api.together.xyz',
)

questions = []
for i, context in enumerate(contexts):
    print(f"Context [{i}]")
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context['context']}\nQuestion:",
            }
        ],
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_tokens=128
    )
    
    question = chat_completion.choices[0].message.content
    if '\n' in question:
        question = question.split('\n')[0]
    if '?' in question:
        question = question.split('?')[0]
    
    questions.append({
        "id": f"synthetic-{str(uuid.uuid4())}",
        "context": context['context'],
        "question": question,
        "answer": "erroneous_question",
        "category": "synthetic_questions",
        "annotator": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    })
    print(questions[-1])
    print("="*40)
    
    if len(questions) % 10 == 0:
        print(f"Generated {len(questions)} questions")
        write_questions_to_jsonl(questions)
    
write_questions_to_jsonl(questions)

