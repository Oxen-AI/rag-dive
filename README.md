# rag-dive

Working code to implement RAG for Question Answering on the SQuAD dataset.

This is the code that goes along with our [Practical ML Dive](https://lu.ma/practicalml)
into RAG.

# Generate context data

```bash
oxen download ox/SQuAD dev.csv
python generate_context.py dev.csv dev_contexts.jsonl
```

# Compute embeddings

TODO: take in CLI args

```bash
python compute_embeddings.py
```

# Download embeddings from Oxen

```bash
oxen download oxbot/SQuAD-Dev-Embed-4 dev_contexts_embeddings.parquet
```

# Setup Chroma

https://docs.trychroma.com/troubleshooting#sqlite

```bash
pip install chromadb==0.4.3
```

```bash
vim ~/.venv_rag/lib/python3.11/site-packages/chromadb/__init__.py
```

Add these few lines...

```python
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```


```python
import chromadb
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="squad_embeddings")
```

Insert all the embeddings into chroma.

TODO: Make cli params work

```bash
python index_into_chroma.py -i embeddings.parquet -o chroma.db
```

# Compute Recall

Figure out how well the embeddings retrieval system works

TODO: Take in N as CLI param

```bash
python compute_recall.py ~/Datasets/Not-In-Context/squad_dev.jsonl chroma-dev.db results.jsonl
```

# Compute Precision

Figure out how well we can extract the answer from the context

```bash
python compute_precision.py -m meta-llama/Llama-2-7b-chat-hf -d ~/Datasets/SQuAD-Context/experiments/dev-recall-3.jsonl -o ~/Datasets/SQuAD-Context/experiments/dev-llama-recall-3-precision-3-shot.jsonl -n 3
```