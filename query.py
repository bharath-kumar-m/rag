import json
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import sys
import subprocess

# Load model offline
model = SentenceTransformer("models/all-MiniLM-L6-v2")

# Load FAISS index and chunks
index = faiss.read_index("faiss_index.idx")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Example query
if len(sys.argv) < 2:
    print("Usage: python query.py <query_string>")
    sys.exit(1)

query = sys.argv[1]
print("Query: ", query)
query_vec = model.encode([query], convert_to_numpy=True)

# Retrieve top 3 relevant chunks
k = 3
D, I = index.search(query_vec.astype(np.float32), k)

print("Top relevant chunks:")
for idx in I[0]:
    print("Chunk: ", chunks[idx])
rag_chunks = "\n".join([chunks[idx] for idx in I[0]])

query = f"""
You are an assistant that analyzes customer server details and the retrieved RAG data (selected from support documentation).

Your tasks are:

Identify the current Service Pack for ProLiant (SPP) bundle used by the customer.
Review the RAG data chunks provided.
Determine the best recommended SPP upgrade for the customer, based on compatibility and relevance.

RAG Data:
{rag_chunks}
"""

payload_dict = {
    "model": "llama3.2:3b",
    "prompt": query
}
command = f'curl -X POST --noproxy localhost -H "Content-Type: application/json" http://localhost:11435/api/generate -d {json.dumps(payload_dict)}'
# Execute the curl command
result = subprocess.run(command, shell=True, capture_output=True, text=True)

# Print the output
if result.returncode == 0:
    print("Response:", result.stdout)
else:
    print("Error:", result.stderr)
