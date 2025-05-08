from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare data
documents = ["Document 1 text...", "Document 2 text..."]
embeddings = model.encode(documents).tolist()
metadata = [{"title": "Doc 1", "url": "http://..."},
            {"title": "Doc 2", "url": "http://..."}]

# Insert into Milvus
collection.insert([list(range(len(documents))), embeddings, metadata])
