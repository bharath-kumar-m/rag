import json
from typing import Dict, List
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
# Load your CSV
df = pd.read_csv('manual_analysis.csv')

# Create enriched text documents for each row
def create_enriched_document(row) -> str:
    """Combine all fields into a rich text description"""
    doc = f"""FWU Error Code: {row['FWU Error Code']}
From Bundle Version: {row['From Bundle Version']}
Target Bundle Version: {row['Target Bundle Version']}
iLO version: {row['iLO version']}
Server Model: {row['Server Model']}
iSUT version: {row['iSUT version']}
COM4VC version: {row['COM4VC version']}
Upgrade type sequence (BMC, SUT, UEFI, Reset, Wait): {row['Upgrade type sequence (BMC, SUT, UEFI, Reset, Wait)']}
Failed Component details: {row['Failed Component details(Filename -> Component name -> From version -> To version)']}
Analysis details: {row['Analysis details']}
Update Type: {row['Update Type']}
Defect ID: {row['Defect ID']}
Customer retry suceeded: {row['Customer retry suceeded']}
"""
    return doc.strip()

# Generate documents
df['enriched_doc'] = df.apply(create_enriched_document, axis=1)

enriched_only = df['enriched_doc'].tolist()
with open('enriched_data.json', 'w') as f:
    json.dump(enriched_only, f, indent=2)


def filter_data(df, filters: Dict) -> pd.DataFrame:
    """Apply exact match filters on categorical data"""
    filtered_df = df.copy()
    
    if 'FWU Error Code' in filters:
        filtered_df = filtered_df[filtered_df['FWU Error Code'] == filters['FWU Error Code']]
    
    if 'From Bundle Version' in filters:
        filtered_df = filtered_df[filtered_df['From Bundle Version'] == filters['From Bundle Version']]
    if 'Target Bundle Version' in filters:
        filtered_df = filtered_df[filtered_df['Target Bundle Version'] == filters['Target Bundle Version']]
    if 'iLO version' in filters:
        filtered_df = filtered_df[filtered_df['iLO version'] == filters['iLO version']]
    if 'Server Model' in filters:
        filtered_df = filtered_df[filtered_df['Server Model'] == filters['Server Model']]

    # Add fuzzy matching if needed
    # if 'version_prefix' in filters:
    #     filtered_df = filtered_df[filtered_df['version'].str.startswith(filters['version_prefix'])]
    
    return filtered_df

# Initialize embedding model
# model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and good for short texts
model = SentenceTransformer("./models/all-MiniLM-L6-v2")
# Embed your enriched documents
embeddings = model.encode(df['enriched_doc'].tolist())

def semantic_search(query: str, filtered_df: pd.DataFrame, filtered_embeddings, top_k=5):
    """Perform semantic search on filtered data"""
    query_embedding = model.encode([query])
    
    # Calculate cosine similarity
    similarities = np.dot(filtered_embeddings, query_embedding.T).flatten()
    
    # Get top-k results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'score': similarities[idx],
            'data': filtered_df.iloc[idx].to_dict()
        })
    
    return results

def query_error_data(query: str, filters: Dict = None, top_k=5):
    """
    Main function to query your error data
    
    Args:
        query: Natural language question
        filters: Dict of exact match filters, e.g., {'version': '2.3.1', 'hardware': 'ModelX'}
        top_k: Number of results to return
    """
    # Step 1: Filter by categorical data
    filtered_df = filter_data(df, filters or {})
    
    if len(filtered_df) == 0:
        return "No results found matching your filters"
    
    # Step 2: Get embeddings for filtered subset
    filtered_indices = filtered_df.index
    filtered_embeddings = embeddings[filtered_indices]
    
    # Step 3: Semantic search on filtered data
    results = semantic_search(query, filtered_df, filtered_embeddings, top_k)
    
    return results

def prepare_context_for_llm(results: List[Dict]) -> str:
    """Format retrieved results as context for your LLM"""
    context = "Relevant error information:\n\n"
    
    for i, result in enumerate(results, 1):
        context += f"Result {i} (Relevance: {result['score']:.2f}):\n"
        context += result['data']['enriched_doc']
        context += "\n\n---\n\n"
    
    return context

query="memory errors during batch processing"
# Example usage:
results = query_error_data(
    query=query,
    filters={'Target Bundle Version': '2.3.1', 'Server Model': 'ModelX'},
    top_k=3
)

# Use with your LLM
context = prepare_context_for_llm(results)
llm_prompt = f"""{context}

User Question: {query}

Please analyze the error information above and provide a detailed answer."""