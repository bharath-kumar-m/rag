import json
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import sys
import subprocess
import requests
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

customer_query = sys.argv[1]
print("Query: ", customer_query)
query_vec = model.encode([customer_query], convert_to_numpy=True)

# Retrieve top 3 relevant chunks
k = 2
D, I = index.search(query_vec.astype(np.float32), k)

# print("Top relevant chunks:")
# for idx in I[0]:
    # print("Chunk: ", chunks[idx])
rag_chunks = "\n".join([chunks[idx] for idx in I[0]])

query = f"""
You are an assistant that analyzes customer server details and the retrieved RAG data (selected from support documentation).
Your tasks are:
Identify the current Service Pack for ProLiant (SPP) bundle used by the customer.
Review the RAG data chunks provided.
Determine the best recommended SPP upgrade for the customer, based on compatibility and relevance.
Result should be directly shown to user, so avoid using chatbox begging and endings
RAG Data:
{rag_chunks}
"Customer Data:
{customer_query}
"""

payload_dict = {
    "model": "llama3.2:3b",
    "prompt": query
}
data=json.dumps(payload_dict)
# Write the data to a temporary file
temp_file = "temp_payload.json"
with open(temp_file, "w") as f:
    f.write(data)
print("Model query: ",data)
command = f'curl -X POST --noproxy localhost -H "Content-Type: application/json" http://localhost:11435/api/generate -d @{temp_file}'
# Execute the curl command
result = subprocess.run(command, shell=True, capture_output=True, text=True)

# Print the output
if result.returncode == 0:
    print("Response:", result.stdout)
else:
    print("Error:", result.stderr)
