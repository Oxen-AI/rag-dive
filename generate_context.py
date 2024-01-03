
import pandas as pd

# iterate over the rows in the csv file
filename = "SQuAD/train.csv"
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
df.to_json("SQuAD/contexts.jsonl", orient="records", lines=True)