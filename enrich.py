import json
from typing import Dict, List
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import subprocess
import sys
import os


# Load your CSV
df = pd.read_csv("manual_analysis.csv")


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


# # Generate documents
# df["enriched_doc"] = df.apply(create_enriched_document, axis=1)

# enriched_only = df["enriched_doc"].tolist()
# with open("enriched_data.json", "w") as f:
#     json.dump(enriched_only, f, indent=2)

# Load enriched documents from JSON file
with open("enriched_data.json", "r") as f:
    enriched_docs = json.load(f)

df["enriched_doc"] = enriched_docs


def filter_data(df, filters: Dict) -> pd.DataFrame:
    """Apply exact match filters on categorical data"""
    filtered_df = df.copy()

    if "FWU Error Code" in filters:
        filtered_df = filtered_df[
            filtered_df["FWU Error Code"] == filters["FWU Error Code"]
        ]

    if "From Bundle Version" in filters:
        filtered_df = filtered_df[
            filtered_df["From Bundle Version"] == filters["From Bundle Version"]
        ]
    if "Target Bundle Version" in filters:
        filtered_df = filtered_df[
            filtered_df["Target Bundle Version"] == filters["Target Bundle Version"]
        ]
    if "iLO version" in filters:
        filtered_df = filtered_df[filtered_df["iLO version"] == filters["iLO version"]]
    if "Server Model" in filters:
        filtered_df = filtered_df[
            filtered_df["Server Model"] == filters["Server Model"]
        ]

    # Add fuzzy matching if needed
    # if 'version_prefix' in filters:
    #     filtered_df = filtered_df[filtered_df['version'].str.startswith(filters['version_prefix'])]

    return filtered_df


# Initialize embedding model
# model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and good for short texts
model = SentenceTransformer("./models/all-MiniLM-L6-v2")
# Embed your enriched documents
embeddings = model.encode(df["enriched_doc"].tolist())


def semantic_search(
    query: str, filtered_df: pd.DataFrame, filtered_embeddings, top_k=5
):
    """Perform semantic search on filtered data"""
    query_embedding = model.encode([query])

    # Calculate cosine similarity
    similarities = np.dot(filtered_embeddings, query_embedding.T).flatten()

    # Get top-k results
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append(
            {"score": similarities[idx], "data": filtered_df.iloc[idx].to_dict()}
        )

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
        return []  # Return empty list instead of string

    # Step 2: Get embeddings for filtered subset
    filtered_indices = filtered_df.index
    filtered_embeddings = embeddings[filtered_indices]

    # Step 3: Semantic search on filtered data
    results = semantic_search(query, filtered_df, filtered_embeddings, top_k)

    return results


def prepare_historical_context_for_llm(results: List[Dict]) -> str:
    """Format retrieved results as context for your LLM"""
    if not results:
        return "No results found matching your filters."

    historical_context = "Historically failure details:\n\n"

    for i, result in enumerate(results, 1):
        score = float(result["score"])  # Ensure it's a float
        historical_context += f"Result {i} (Relevance: {score:.2f}):\n"
        historical_context += result["data"]["enriched_doc"]
        historical_context += "\n\n---\n\n"

    return historical_context


def prepare_current_failure_context_for_llm() -> str:
    """Format retrieved results as context for your LLM"""
    current_failure_context = "Failure AHS logs from ilo:\n\n"
    ahs = """   
2026/01/07 12:47:55 | Firmware Flash | Error downloading component from default Url. err = -9\n
2026/01/07 12:47:55 | Firmware Flash | DSE : Failed to copy the component cp068954.exe into NAND, error code returned is -9.
2026/01/07 12:47:55 | Firmware Flash | DSE : calling msg_BundleUpdateComponentDownloadFailed for InternalError : msgindex = 7 message = cp068954.exe
2026/01/07 12:47:55 | Firmware Flash | DSE : Component download to NAND failed : stopping bundle update
2026/01/07 12:47:55 | Firmware Flash | DSE : --------- Completed Report --------
2026/01/07 12:47:55 | Firmware Flash | DSE : comp cnt = 7 cmd cnt = 1 sppver =  status = 0\n
2026/01/07 12:47:55 | Firmware Flash | DSE : COMP 0 : name = P14.4.731.5_header.pldm.fwpkg ver = 14.4.731.5 dirty flag = 0 status = 5 bundleOp:3 UpdateRequest:0
2026/01/07 12:47:55 | Firmware Flash | DSE : COMP 1 : name = U54_2.70_10_31_2025.fwpkg ver = 2.70_10-31-2025 dirty flag = 0 status = 5 bundleOp:3 UpdateRequest:0
2026/01/07 12:47:55 | Firmware Flash | DSE : COMP 2 : name = bcm235.1.164.14.fwpkg ver = 235.1.164.14 dirty flag = 0 status = 5 bundleOp:3 UpdateRequest:0
2026/01/07 12:47:55 | Firmware Flash | DSE : COMP 3 : name = cp068887.exe ver = 6.4.0.0 dirty flag = 0 status = 5 bundleOp:3 UpdateRequest:0
2026/01/07 12:47:55 | Firmware Flash | DSE : COMP 4 : name = cp068954.exe ver =  dirty flag = 0 status = 4 bundleOp:1 UpdateRequest:0
2026/01/07 12:47:55 | Firmware Flash | DSE : COMP 5 : name = cp069012.exe ver =  dirty flag = 0 status = 5 bundleOp:1 UpdateRequest:0
2026/01/07 12:47:55 | Firmware Flash | DSE : COMP 6 : name = cp068067.exe ver =  dirty flag = 0 status = 5 bundleOp:1 UpdateRequest:0
2026/01/07 12:47:55 | Firmware Flash | DSE : CMD 0 : name = Server Reset Task 671800460 updateble by = RuntimeAgent cmd = ResetServer status = 0\n
2026/01/07 12:47:55 | Firmware Flash | DSE : final stat : size = 225075074 : download time = 484962978 : installation time = 0
2026/01/07 12:47:55 | Firmware Flash | DseStateManager: State transition completed from Idle -> Bundle Update Complete
2026/01/07 12:47:58 | ILO event logs | Failed to download the Bundle Update component "cp068954.exe" ACTION: Retry Bundle Update after verifying the network settings and location of the component.
2026/01/07 12:47:59 | ILO event logs | REST - Task Id 19 (UpdateService.SimpleUpdate) has changed state to exception.


    """
    current_failure_context += ahs
    return current_failure_context


def prepare_instruction_context_for_llm() -> str:
    """Provide instructions for the LLM"""
    return """
        You are a Firmware Update Failure Triage Engine.
        You must strictly follow ALL rules below. Any violation makes the response invalid.

        GENERAL RULES:
        - Do NOT speculate, assume, or guess.
        - Do NOT introduce causes not directly supported by log evidence.
        - Only analyze information explicitly present in the input logs.
        - If a conclusion cannot be directly traced to a log line, do NOT state it.
        - If evidence is insufficient, explicitly state: "INSUFFICIENT_EVIDENCE".
        - Prefer deterministic statements. Do NOT use probabilistic words such as: "likely", "possibly", "appears", "may", "could".

        CONTENT RULES:
        - Do NOT suggest fixes or corrective actions unless a root cause is identified AND supported by log evidence.
        - Do NOT use generic explanations.
        - Do NOT expand or explain abbreviations.
        - Do NOT include or reference logs containing "fwrepo_open".
        - Always provide the failed component or file names in the output
        - Always replace the term "webserver" with "COM".
        - Do NOT display or mention: "Error Codes", "**Hint**", "Hints", or "Note".
        - Every major statement in the Summary and Conclusion MUST include at least one direct log reference.
        - A log reference must be an exact log line or a clearly identifiable substring from the logs.
        - Log references must be enclosed in double quotes.
        - Do NOT invent log references.
        - If a statement cannot be supported with a log reference, do NOT include that statement.

        STRUCTURE RULES:
        - The response MUST contain exactly two sections, in this order:
            1. Analysis
            2. Conclusion

        - No additional sections or headings are allowed.

        ANALYSIS SECTION RULES:
        - The Analysis section MUST be based only on the provided logs.
        - All analysis statements MUST reference AHS log lines directly.
        - If analysis cannot be performed due to insufficient data,
        output exactly: "No analysis due to insufficient data".

        CONCLUSION:
        - The Conclusion section may ONLY contain:
        - A confirmed outcome supported directly by log evidence
        OR
        - "No conclusion due to insufficient evidence"

        SUMMARY RULES:
        - Limit the overall response to approximately 15 concise lines.
        - Include log references wherever possible.

        Your goal is to analyze and classify the failure deterministically.
        Do not chat. Do not explain your reasoning process.

    """


def call_llm_api(query):
    """Call the local LLM API with the query."""
    payload_dict = {
        "model": MODEL,
        "prompt": query,
        "stream": True,
        "options": {
            "temperature": 0.3,  # Lower temperature for more focused output
            "top_p": 0.9,
            "num_predict": 2000,  # Limit output length
        },
    }
    data = json.dumps(payload_dict)

    # Write the data to a temporary file
    temp_file = os.path.abspath("temp_payload.json")
    with open(temp_file, "w") as f:
        f.write(data)

    print(f"\n{'='*60}")
    print(f"Querying {MODEL} LLM ...")
    print(f"{'='*60}\n")
    command = f'curl -X POST --noproxy localhost -H "Content-Type: application/json" http://localhost:11435/api/generate -d @{temp_file}'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    return result


def parse_llm_response(result):
    """Parse the response from the LLM API."""
    response_text = ""

    if result.strip():
        for line in result.strip().split("\n"):
            try:
                entry = json.loads(line)
                if "response" in entry:
                    response_text += entry["response"]
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                print(f"Problematic line: {line}")

    return response_text


MODEL = "qwen2.5:14b"


def main():
    """Main function to process the customer query."""
    if len(sys.argv) < 2:
        # print("Usage: python query.py <query_string>")
        sys.exit(1)

    customer_query = sys.argv[1]
    query = "FWU Error Code FWE-108"
    # Example usage:
    # results = query_error_data(
    #     query=query, filters={"FWU Error Code": "FWE-108"}, top_k=3
    # )

    # Use with your LLM
    # historical_context = prepare_historical_context_for_llm(results)
    # print(f"\nLLM histrical embeddings context:\n{historical_context}")
    current_failure_context = prepare_current_failure_context_for_llm()
    # print(f"\nCurrent failure context:\n{current_failure_context}")
    llm_instruction = prepare_instruction_context_for_llm()
    # print(f"\nLLM instruction:\n{llm_instruction}")

    # Build better structured prompt
    llm_prompt = f"""{llm_instruction} \n {current_failure_context}"""

    # Call LLM API
    result = call_llm_api(llm_prompt)

    # Parse response
    response_text = parse_llm_response(result.stdout)

    # Display results
    print(f"\n{'='*60}")
    print("FIRMWARE UPDATE FAILURE REPORT")
    print(f"{'='*60}\n")
    print(response_text)
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()


# Ahs, SUT STATIC LOGS.
