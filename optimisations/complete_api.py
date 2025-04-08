import os
import sys
import asyncio
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as gai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
import logging
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI()
class QueryRequest(BaseModel):
    user_prompt: str

# Load env vars
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # Base Gemini
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
QDRANT_DB_URL = os.environ.get("QDRANT_DB_URL")

# Base Gemini setup
if GEMINI_API_KEY:
    gai.configure(api_key=GEMINI_API_KEY)
    base_gemini_client = gai.GenerativeModel('gemini-1.5-flash')
    logger.info("Base Gemini client initialized")
else:
    base_gemini_client = None
    logger.warning("No GEMINI_API_KEY - /rag will use Groq only")

# RAG function (base Gemini or Groq fallback)
async def generate_response(user_prompt: str, max_passes: int = 3, model: int = 0) -> str:
    qdrant_client = QdrantClient(QDRANT_DB_URL)
    embedder = SentenceTransformer("BAAI/bge-small-en")
    groq_client = Groq(api_key=GROQ_API_KEY)

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
            if model == 0 or not base_gemini_client:  # Groq fallback
                eval_response = await asyncio.to_thread(
                    groq_client.chat.completions.create,
                    temperature=0.7,
                    n=1,
                    model="deepseek-r1-distill-llama-70b",
                    messages=eval_messages,
                    max_tokens=1000
                )
                eval_response = eval_response.choices[0].message.content
            else:  # Base Gemini
                eval_response = await asyncio.to_thread(
                    base_gemini_client.generate_content,
                    eval_prompt
                )
                eval_response = eval_response.text
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
        if model == 0 or not base_gemini_client:
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
                base_gemini_client.generate_content,
                final_prompt
            )
            final_response = final_response.text
    except Exception as e:
        logger.error(f"Final error: {str(e)}")
        raise

    return final_answer or final_response

# Routes
@app.post("/gemini")
async def gemini_direct(query: QueryRequest):
    if not base_gemini_client:
        raise HTTPException(status_code=500, detail="Base Gemini unavailable - provide GEMINI_API_KEY")
    try:
        system_prompt = "You are a precise coding assistant."
        full_prompt = f"{system_prompt}\n\nUser query: {query.user_prompt}"
        response = await asyncio.to_thread(base_gemini_client.generate_content, full_prompt)
        return {"response": response.text}
    except Exception as e:
        logger.error(f"Gemini error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

@app.post("/rag")
async def rag_generate(query: QueryRequest):
    try:
        response = await generate_response(query.user_prompt, max_passes=3, model=1 if base_gemini_client else 0)
        return {"response": response}
    except Exception as e:
        logger.error(f"RAG error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)