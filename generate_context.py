
import pandas as pd
import sys

# make sure we have an input file and output file from the command line
if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <input_file> <output_file>")
    sys.exit(1)

# iterate over the rows in the csv file
filename = sys.argv[1]
df = pd.read_csv(filename)

contexts = {}

print("Computing context...")
for index, row in df.iterrows():
    context = row["context"]
    if context not in contexts:
        contexts[context] = []
    
    contexts[context].append(row["id"])

print("Creating dataframe...")
data = []
for context in contexts:
    data.append({
        "context": context,
        "question_ids": contexts[context]
    })

# write new jsonl file
df = pd.DataFrame(data)

print("Writing to file...")
df.to_json(sys.argv[2], orient="records", lines=True)