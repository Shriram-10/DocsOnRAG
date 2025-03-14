import os
import torch
from sentence_transformers import SentenceTransformer
from base_code.text_extraction import extract_text_from_pdf, chunk_text
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, VectorParams, Distance

# Load the embedding model
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-small-en", device=device)

# Load Qdrant client
db_url = os.environ.get("QDRANT_DB_URL")  # Ensure QDRANT_DB_URL is set
client = QdrantClient(db_url)

# Base directories
pdf_base_folder = "./python_docs"
batch_size = 1000  # Adjust based on memory

# Name of the Qdrant collection for Python docs
collection_name = "python_docs"

# Ensure the collection exists
if not client.collection_exists(collection_name=collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# Get all subdirectories (each representing a Python version)
versions = [d for d in os.listdir(pdf_base_folder) if os.path.isdir(os.path.join(pdf_base_folder, d))]

all_text_chunks = []
all_metadata = []
all_embeddings = []

for version in versions:
    print(f"\n📂 Processing Python Docs for Version: {version}\n")

    pdf_folder = os.path.join(pdf_base_folder, version)

    # Extract text from PDFs
    for root, _, files in os.walk(pdf_folder):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"📄 Processing: {pdf_path}")

                try:
                    text = extract_text_from_pdf(pdf_path)
                    text_chunks = chunk_text(text)

                    for index, chunk in enumerate(text_chunks):
                        all_text_chunks.append(chunk)
                        all_metadata.append({
                            "text": chunk,
                            "source": pdf_path,
                            "page_number": index,
                            "version": version  # Store version as metadata
                        })
                except Exception as e:
                    print(f"❌ Error processing {pdf_path}: {e}")

# Generate embeddings in batches
for i in tqdm(range(0, len(all_text_chunks), batch_size), desc=f"Generating Embeddings"):
    batch = all_text_chunks[i : i + batch_size]
    embeddings = model.encode(batch, convert_to_tensor=False)  # Convert to numpy
    all_embeddings.extend(embeddings)

if all_embeddings:
    # Prepare data for upload
    points = [
        PointStruct(
            id=i,  
            vector=all_embeddings[i].tolist(),
            payload=all_metadata[i]  # Attach metadata
        )
        for i in range(len(all_embeddings))
    ]

    # Upload entire dataset to Qdrant
    client.upload_collection(collection_name=collection_name, vectors=[p.vector for p in points], payload=[p.payload for p in points])

    print(f"🚀 Embeddings uploaded to Qdrant for {collection_name}!\n")
else:
    print(f"⚠️ No embeddings generated, skipping save.\n")

print("🎉 All Python docs processed and stored in Qdrant successfully!")
