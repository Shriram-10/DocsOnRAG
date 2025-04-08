import os
import re
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as gai
from dotenv import load_dotenv

# Load env vars
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
QDRANT_DB_URL = os.environ.get("QDRANT_DB_URL", "http://localhost:6333")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

# Initialize clients
gai.configure(api_key=GEMINI_API_KEY)
gemini_client = gai.GenerativeModel('gemini-1.5-flash')
qdrant_client = QdrantClient(QDRANT_DB_URL)
embedder = SentenceTransformer("BAAI/bge-small-en")

def generate_response(user_prompt: str, max_passes: int = 3) -> str:
    collection_name = "python_docs"
    all_retrieved_texts = set()
    query = user_prompt
    pass_count = 0
    final_answer = None

    while pass_count < max_passes:
        pass_count += 1
        print(f"\nPass {pass_count}")
        query_embedding = embedder.encode(query).tolist()
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=20,
            score_threshold=0.7,
            with_payload=True
        )
        retrieved_texts = set()
        for hit in search_results:
            if "text" in hit.payload and hit.payload["text"] not in all_retrieved_texts:
                retrieved_texts.add(hit.payload["text"])
                all_retrieved_texts.add(hit.payload["text"])

        retrieved_texts_str = "\n".join(retrieved_texts)
        print(f"Retrieved: {retrieved_texts_str[:100]}...")

        summary = " ".join([text[:200] for text in retrieved_texts])
        eval_prompt = f"""You are a precise coding assistant.
        Summary of retrieved docs:\n{summary}\n\n
        Full retrieved docs:\n{retrieved_texts_str}\n\n
        Does this provide enough info to answer '{user_prompt}' fully? If yes, provide the answer.
        If no, explain what’s missing and suggest a new query.
        Format:
        - Sufficient: Yes/No
        - Answer: (if Yes, the answer; if No, blank)
        - Missing: (if No, e.g., 'Syntax variants')
        - New Query: (if No, a refined query)"""

        eval_response = gemini_client.generate_content(eval_prompt).text
        print(f"Eval: {eval_response[:100]}...")

        sufficient = "Sufficient: Yes" in eval_response
        if sufficient:
            answer_match = re.search(r"Answer: ([\s\S]+)", eval_response)
            if answer_match:
                final_answer = answer_match.group(1).strip()

        if not sufficient:
            query_match = re.search(r"New Query: ([\s\S]+)", eval_response)
            query = query_match.group(1).strip() if query_match else f"Detailed {user_prompt}"

    final_prompt = f"""You are a coding assistant with access to documentation.
    Based on all retrieved Python docs:\n\n{'\n'.join(all_retrieved_texts)}\n\n
    Answer: {user_prompt}"""
    final_response = gemini_client.generate_content(final_prompt).text
    return final_answer or final_response

# Test
queries = [
    "What is the os module in Python?",
    "How do I use sys.argv?",
    "Write a for loop example."
]

for query in queries:
    try:
        response = generate_response(query)
        print(f"\nQuery: {query}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error for '{query}': {str(e)}")

print("Done!")