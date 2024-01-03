# rag-dive

Working code to implement RAG for Question Answering on the SQuAD dataset.

This is the code that goes along with our [Practical ML Dive](https://lu.ma/practicalml)
into RAG.

# Generate context data

```bash
oxen clone https://hub.oxen.ai/ox/SQuAD --shallow
cd SQuAD
oxen remote download train.csv
cd ..
python generate_context.py
```

# Embedding Server

```bash
modal serve embeddings.py
```

# Compute embeddings

```bash
modal run embeddings.py
```

# Download embeddings from Oxen

```bash
oxen download oxbot/SQuAD-Ctx-Embeddings embeddings.parquet
```

# Setup Chroma

```bash
pip install chromadb
```


```python
import chromadb
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="squad_embeddings")
```



