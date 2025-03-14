from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

collection_name = "python_docs"
model = SentenceTransformer("BAAI/bge-small-en")
load_dotenv()
db_url = os.environ.get("QDRANT_DB_URL")
client = QdrantClient(db_url)
query = "How does async work in Python?"  # Example user query
query_embedding = model.encode(query).tolist()  # Convert query to vector

search_results = client.query_points(
    collection_name=collection_name,
    query=query_embedding,
    limit=5,  # Get top 5 relevant chunks
    with_payload=True  # Retrieve the associated text chunks
)

# Extract text from payload correctly
print("Search Results:", search_results)

# print("\n".join(retrieved_texts))  # Display retrieved chunks