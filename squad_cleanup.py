
import json
import sys
import ast

# Require two command line parameters
if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <input_file> <output_file>")
    sys.exit(1)

# Read in jsonl file
with open(sys.argv[1], 'r') as f:
    data = [json.loads(line) for line in f]
    
# process data
results = []
for d in data:
    # print(d)
    answers = ast.literal_eval(d['answers'])
    if len(answers) > 0:
        answer = answers[0]['text']
    else:
        answer = "not_in_context"

    results.append({
        "id": f"{d['id']}",
        "context": d["context"],
        "question": d["question"],
        "answer": answer,
        # "category": "squad_questions",
        # "annotator": "squad",
    })

# Write out jsonl file
with open(sys.argv[2], 'w') as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

