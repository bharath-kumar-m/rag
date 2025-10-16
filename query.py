from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# Load model offline
model = SentenceTransformer("models/all-MiniLM-L6-v2")

# Load FAISS index and chunks
index = faiss.read_index("faiss_index.idx")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Example query
query = "Give me the release note summary for SPP 2025.01.00.00"
query_vec = model.encode([query], convert_to_numpy=True)

# Retrieve top 3 relevant chunks
k = 3
D, I = index.search(query_vec.astype(np.float32), k)

print("Top relevant chunks:")
for idx in I[0]:
    print("—", chunks[idx])
