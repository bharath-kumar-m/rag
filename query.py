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
You are an assistant that analyzes customer server details. 
Use ONLY the information provided in the RAG data internally to determine your answer, but do NOT mention the RAG data in your response. 
Tasks:
1. Identify the current Service Pack for ProLiant (SPP) bundle.
2. Recommend the best SPP upgrade for the customer, if available.
3. Provide the answer **directly and clearly**, as if you are giving instructions to the customer. 
4. Do NOT invent or hallucinate information. If the required information is missing, respond with "Information not available."
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

# Check if command succeeded
if result.returncode == 0:
    # Get the response as a single string
    response_text = result.stdout.replace("\n", " ").strip()
    
    # Save to a file
    output_file = "customer_response.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(response_text)
    
    print(f"Response saved to {output_file}")
else:
    print("Error:", result.stderr)