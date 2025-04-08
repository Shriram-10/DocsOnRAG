import asyncio
import re
import uvicorn
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
import logging
import os
from dotenv import load_dotenv

load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration
API_KEY = os.environ.get("FINETUNED_GEMINI_API_KEY")
API_URL = "https://generativelanguage.googleapis.com/v1beta/tunedModels/pythonchatbotdataset-6scpyuehcnr5:generateContent"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # Replace with your Groq API key
QDRANT_DB_URL = 'http://localhost:6333'  # Adjust if different

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)

# Function to make curl-equivalent request
def call_gemini_api(prompt: str) -> str:
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }
    try:
        response = requests.post(
            f"{API_URL}?key={API_KEY}",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        # Extract the text from the response (adjust based on actual response structure)
        response_data = response.json()
        # Assuming response structure is {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
        return response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    except Exception as e:
        logger.error(f"Gemini API call failed: {str(e)}")
        raise

# Pydantic model for request
class QueryRequest(BaseModel):
    user_prompt: str

# RAG function with tuned Gemini or Groq fallback
async def generate_response(user_prompt: str, max_passes: int = 3, model: int = 0) -> str:
    qdrant_client = QdrantClient(QDRANT_DB_URL)
    embedder = SentenceTransformer("BAAI/bge-small-en")
    
    collection_name = "python_docs"
    all_retrieved_texts = set()
    query = user_prompt
    pass_count = 0
    final_answer = None
    prev_retrieved_count = 0

    while pass_count < max_passes:
        pass_count += 1
        query_embedding = embedder.encode(query).tolist()
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=20,
            score_threshold=0.7,
            with_payload=True
        )
        retrieved_texts = set()
        retrieved_pages = []
        for hit in search_results:
            if "text" in hit.payload and hit.payload["text"] not in all_retrieved_texts:
                retrieved_texts.add(hit.payload["text"])
                all_retrieved_texts.add(hit.payload["text"])
                page_number = hit.payload.get("page_number")
                if page_number is not None:
                    retrieved_pages.append(page_number)
        for page in retrieved_pages:
            surrounding_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=10,
                with_payload=True
            )
            for doc in surrounding_results:
                if "text" in doc.payload and doc.payload["text"] not in all_retrieved_texts:
                    retrieved_texts.add(doc.payload["text"])
                    all_retrieved_texts.add(doc.payload["text"])

        retrieved_texts_str = "\n".join(retrieved_texts)
        summary = " ".join([text[:200] for text in retrieved_texts])
        eval_prompt = f"""You are an AI assistant tasked with answering: '{user_prompt}'.
        Summary of retrieved docs:\n{summary}\n\n
        Full retrieved docs:\n{retrieved_texts_str}\n\n
        Does this provide enough info to answer fully? If yes, provide the answer.
        If no, explain what’s missing and suggest a new query.
        Format:
        - Sufficient: Yes/No
        - Answer: (if Yes, the answer; if No, blank)
        - Missing: (if No, e.g., 'Syntax variants')
        - New Query: (if No, a refined query)"""

        eval_messages = [{"role": "system", "content": "You are a precise coding assistant."}, {"role": "user", "content": eval_prompt}]
        eval_response = None
        try:
            if model == 0:  # Groq
                eval_response = await asyncio.to_thread(
                    groq_client.chat.completions.create,
                    temperature=0.7,
                    n=1,
                    model="deepseek-r1-distill-llama-70b",
                    messages=eval_messages,
                    max_tokens=1000
                )
                eval_response = eval_response.choices[0].message.content
            else:  # Gemini via curl
                eval_response = await asyncio.to_thread(
                    call_gemini_api,
                    eval_prompt
                )
        except Exception as e:
            logger.error(f"Eval error (Pass {pass_count}): {str(e)}")
            raise

        sufficient = "Sufficient: Yes" in eval_response
        if sufficient:
            answer_match = re.search(r"Answer: ([\s\S]+)", eval_response)
            if answer_match:
                final_answer = answer_match.group(1).strip()

        if not sufficient:
            query_match = re.search(r"New Query: ([\s\S]+)", eval_response)
            query = query_match.group(1).strip() if query_match else f"Detailed syntax and advanced examples of {user_prompt}"
            if pass_count > 1 and len(retrieved_texts) < prev_retrieved_count * 1.2:
                break
            prev_retrieved_count = len(retrieved_texts)

    final_prompt = f"""Based on all retrieved Python docs:\n\n{'\n'.join(all_retrieved_texts)}\n\nAnswer: {user_prompt}"""
    final_messages = [{"role": "system", "content": "You are a coding assistant with access to documentation."}, {"role": "user", "content": final_prompt}]
    final_response = None
    try:
        if model == 0:
            final_response = await asyncio.to_thread(
                groq_client.chat.completions.create,
                temperature=1.1,
                n=1,
                model="deepseek-r1-distill-llama-70b",
                messages=final_messages,
                max_tokens=2000
            )
            final_response = final_response.choices[0].message.content
        else:
            final_response = await asyncio.to_thread(
                call_gemini_api,
                final_prompt
            )
    except Exception as e:
        logger.error(f"Final error: {str(e)}")
        raise

    return final_answer or final_response

# Routes
@app.post("/tuned-gemini")
async def tuned_gemini_direct(query: QueryRequest):
    try:
        system_prompt = "You are a precise coding assistant trained to predict the next number or word in a sequence."
        full_prompt = f"{system_prompt}\n\nUser query: {query.user_prompt}"
        response = await asyncio.to_thread(
            call_gemini_api,
            full_prompt
        )
        return {"response": response}
    except Exception as e:
        logger.error(f"Tuned Gemini error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tuned Gemini error: {str(e)}")

@app.post("/tuned-gemini-rag")
async def tuned_gemini_rag(query: QueryRequest):
    try:
        response = await generate_response(query.user_prompt, max_passes=3, model=1)
        return {"response": response}
    except Exception as e:
        logger.error(f"Tuned Gemini RAG error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tuned Gemini RAG error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)