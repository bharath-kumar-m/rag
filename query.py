import json
import pickle
import subprocess
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os


def load_model_and_data():
    """Load the sentence transformer model, FAISS index, and chunks."""
    # Load model offline
    model = SentenceTransformer("models/all-MiniLM-L6-v2")
    
    # Load FAISS index and chunks
    index = faiss.read_index("faiss_index.idx")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    
    return model, index, chunks

def search_relevant_chunks(model, index, chunks, query_text, k=5):
    """Search for relevant chunks using the query."""
    query_vec = model.encode([query_text], convert_to_numpy=True)
    D, I = index.search(query_vec.astype(np.float32), k)
    rag_chunks = "\n".join([chunks[idx] for idx in I[0]])
    return rag_chunks


def create_query_prompt(rag_chunks, customer_query):
    """Create the query prompt for the LLM."""
    return f"""
You are a system that analyzes customer server details.
Your job is to generate a short, factual report for the customer.

Rules:
1. Use ONLY the information from the RAG data. Do NOT mention or refer to it in your output.
2. Do NOT ask questions or include conversational language.
3. Respond in a clear, formal, and instructional tone suitable for a customer-facing report.
4. If any information is missing, write: "Information not available."
5. Structure your answer as:
6. Add recc on why updating is good if there's any: <short one-line advice>

Current SPP Bundle: <value>
Recommended Upgrade: <value >
Recommendation: <short one-line advice>
Below is the information, use it to answer the query.
RAG Data:
{rag_chunks}
Customer Data:
{customer_query}
"""


def call_llm_api(query):
    """Call the local LLM API with the query."""
    payload_dict = {
        "model": "llama3.2:3b",
        "prompt": query
    }
    data = json.dumps(payload_dict)
    
    # Write the data to a temporary file
    temp_file = os.path.abspath("temp_payload.json")
    with open(temp_file, "w") as f:
        f.write(data)
    
    print("Model query:", data)
    
    command = f'curl -X POST --noproxy localhost -H "Content-Type: application/json" http://localhost:11435/api/generate -d @{temp_file}'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    return result


def parse_llm_response(result):
    """Parse the response from the LLM API."""
    response_text = ""
    
    if result.strip():
        for line in result.strip().split('\n'):
            try:
                entry = json.loads(line)
                if "response" in entry:
                    response_text += entry["response"]
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                print(f"Problematic line: {line}")
    
    return response_text


def save_response(response_text, filename="customer_response.txt"):
    """Save the response to a file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(response_text)
    print(f"Response saved to {filename}")


def main():
    """Main function to process the customer query."""
    if len(sys.argv) < 2:
        print("Usage: python query.py <query_string>")
        sys.exit(1)

    customer_query = sys.argv[1]
    print("Query:", customer_query)
    
    # Load model and data
    model, index, chunks = load_model_and_data()
    
    # Search for relevant chunks
    rag_chunks = search_relevant_chunks(model, index, chunks, customer_query)
    
    # Create query prompt
    query = create_query_prompt(rag_chunks, customer_query)
    
    # Call LLM API
    result = call_llm_api(query)
    
    # Parse response
    response_text = parse_llm_response(result.stdout)


if __name__ == "__main__":
    main()
