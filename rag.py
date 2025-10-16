from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# 1. Load model (online machine)
model = SentenceTransformer("./models/all-MiniLM-L6-v2")

# 2. Read firmware release notes
with open("release_notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 3. Split text into chunks
def chunk_text(text, chunk_size=500):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

chunks = list(chunk_text(text))

# 4. Generate embeddings
embeddings = model.encode(chunks, convert_to_numpy=True)

# 5. Create FAISS index
d = embeddings.shape[1]  # embedding dimension
index = faiss.IndexFlatL2(d)
index.add(embeddings.astype(np.float32))

# 6. Save FAISS index and chunks
faiss.write_index(index, "faiss_index.idx")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

# Optional: save model locally for offline server
model.save("models/all-MiniLM-L6-v2")
