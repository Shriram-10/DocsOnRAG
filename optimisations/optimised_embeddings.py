import os
import sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base_code.text_extraction import extract_text_from_pdf, chunk_text
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, VectorParams, Distance
import time

# Load embedding model
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
model = SentenceTransformer("BAAI/bge-small-en", device=device)

# Load Qdrant client
db_url = os.environ.get("QDRANT_DB_URL")
client = QdrantClient(db_url)

# Base directories
pdf_base_folder = "./python_docs"
batch_size = 1000  # Embedding/Qdrant batch size
collection_name = "python_docs"

# Ensure collection exists
if not client.collection_exists(collection_name=collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
else:
    print(f"Collection '{collection_name}' already exists in Qdrant")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("Collection reset")
    print(f"Total points in Qdrant: {client.count(collection_name=collection_name).count}")

# Process docs
versions = [d for d in os.listdir(pdf_base_folder) if os.path.isdir(os.path.join(pdf_base_folder, d))]
all_points = []
total_chunks_embedded = 0  # Track total chunks embedded

start_total = time.time()
for version in versions:
    print(f"\n📂 Processing Python Docs for Version: {version}\n")
    pdf_folder = os.path.join(pdf_base_folder, version)
    for root, _, files in os.walk(pdf_folder):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"📄 Processing: {pdf_path}")
                try:
                    start_extract = time.time()
                    text = extract_text_from_pdf(pdf_path)
                    chunks = chunk_text(text)
                    extract_time = time.time() - start_extract
                    print(f"Extraction + chunking: {extract_time:.2f}s, Chunks generated: {len(chunks)}")
                    
                    start_embed = time.time()
                    pdf_chunks_embedded = 0  # Track chunks embedded for this PDF
                    for i in range(0, len(chunks), batch_size):
                        batch_chunks = chunks[i:i + batch_size]
                        texts = [chunk for chunk in batch_chunks]  # Adjust if chunk_text returns dict
                        start_batch = time.time()
                        embeddings = model.encode(texts, convert_to_tensor=True).cpu().numpy()
                        batch_time = time.time() - start_batch
                        batch_chunks_embedded = len(embeddings)
                        pdf_chunks_embedded += batch_chunks_embedded
                        total_chunks_embedded += batch_chunks_embedded
                        print(f"Batch {i//batch_size + 1} embedded {batch_chunks_embedded} chunks in {batch_time:.2f}s")
                        for idx, embedding in enumerate(embeddings):
                            all_points.append(PointStruct(
                                id=len(all_points),
                                vector=embedding.tolist(),
                                payload={
                                    "text": batch_chunks[idx],
                                    "source": pdf_path,
                                    "page_number": (i + idx) // 2 + 1,
                                    "chunk_index": f"{version}_{file}_{i + idx}",
                                    "version": version
                                }
                            ))
                        if len(all_points) >= batch_size:
                            client.upload_collection(
                                collection_name=collection_name,
                                vectors=[p.vector for p in all_points],
                                payload=[p.payload for p in all_points]
                            )
                            print(f"Uploaded {len(all_points)} chunks to Qdrant")
                            all_points.clear()
                    embed_time = time.time() - start_embed
                    print(f"Embedding completed: {embed_time:.2f}s, Total chunks embedded for {file}: {pdf_chunks_embedded}")
                except Exception as e:
                    print(f"❌ Error: {e}")

# Final upload
if all_points:
    client.upload_collection(
        collection_name=collection_name,
        vectors=[p.vector for p in all_points],
        payload=[p.payload for p in all_points]
    )
    print(f"Final upload: {len(all_points)} chunks to Qdrant")

print(f"Total time: {time.time() - start_total:.2f}s")
print(f"Total chunks embedded: {total_chunks_embedded}")
print(f"Total points in Qdrant: {client.count(collection_name=collection_name).count}")
print("🎉 All Python docs processed and stored in Qdrant successfully!")