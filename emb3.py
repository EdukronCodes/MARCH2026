import os
import json
import numpy as np
import chromadb
from chromadb import PersistentClient
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------------------
# Initialize Chroma Client
# ----------------------------------------
client = PersistentClient(path="./chroma_data")

collection = client.get_or_create_collection(
    name="tickets_simple",
    metadata={"hnsw:space": "cosine"}
)

# ----------------------------------------
# Load Ticket Data
# ----------------------------------------
data_folder = "./data"

documents = []
metadatas = []
ids = []

for filename in os.listdir(data_folder):
    if filename.endswith(".json"):
        with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as f:
            ticket = json.load(f)

        doc_id = str(ticket.get("ticket_id", filename))
        text = f"{ticket.get('title', '')} {ticket.get('description', '')}"

        documents.append(text)
        ids.append(doc_id)

        metadatas.append({
            "ticket_id": ticket.get("ticket_id"),
            "status": ticket.get("status"),
            "priority": ticket.get("priority"),
            "category": ticket.get("category"),
            "department": ticket.get("department")
        })

# ----------------------------------------
# Generate TF-IDF Embeddings
# ----------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
doc_embeddings = vectorizer.fit_transform(documents).toarray()

# ----------------------------------------
# Add to ChromaDB
# ----------------------------------------
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids,
    embeddings=doc_embeddings.tolist()
)

print(f"✅ Inserted {len(documents)} tickets")

# ----------------------------------------
# Query
# ----------------------------------------
query = "password reset and login failure"

query_embedding = vectorizer.transform([query]).toarray()

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=5
)

# ----------------------------------------
# Display Results
# ----------------------------------------
print("\n🔍 Search Results\n")

for i in range(len(results["ids"][0])):
    print(f"Rank #{i+1}")
    print("Ticket ID:", results["ids"][0][i])
    print("Text:", results["documents"][0][i])
    print("Metadata:", results["metadatas"][0][i])
    print("Distance:", results["distances"][0][i])
    print("-" * 50)

print("\u26A0\uFE0F Warning: Invalid input detected")
