# vector_store.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []

    def build(self, texts):
        self.chunks = texts
        embeddings = model.encode(texts, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def query(self, prompt, top_k=5):
        prompt_emb = model.encode([prompt], convert_to_numpy=True)
        D, I = self.index.search(prompt_emb, top_k)
        return [self.chunks[i] for i in I[0]]
