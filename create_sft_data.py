
import sys
import json

# Require input and output file from args
if len(sys.argv) != 3:
    print("Usage: python create_sft_data.py <input_file> <output_file>")
    sys.exit(1)

# Read in jsonl file
with open(sys.argv[1], "r") as f:
    data = [json.loads(line) for line in f]

# Process data into format for SFT
sft_data = []

for d in data:
    text = f"""<text>
    <context>
        {d["context"]}
    </context>
    <question>
        {d["question"]}
    </question>
    <answer>
        {d["answer"]}
    </answer>
</text>
"""
    sft_data.append({
        "text": text
    })

# Write to output file
with open(sys.argv[2], "a") as f:
    for d in sft_data:
        f.write(json.dumps(d) + "\n")
