import os
import sys
import re
import asyncio
from fastapi import FastAPI
import uvicorn
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
import google.generativeai as gai
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def extract_generated_text(response: str) -> str:
    match = re.search(r"(?:</think>\s*|Answer:\s*)([\s\S]*)", response.strip())
    if match:
        print("REGEX MATCHING:", match)
        return match.group(1).strip()
    cleaned = re.sub(r"<think>[\s\S]*?</think>\s*", "", response.strip())
    print("REGEX MATCHING: None, cleaned output")
    return cleaned.strip() or response.strip()

async def generate_response(user_prompt: str, max_passes: int = 3, model: int = 0) -> str:
    load_dotenv()
    db_url = os.environ.get("QDRANT_DB_URL")
    groq_api_key = os.environ.get("GROQ_API_KEY")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    print("DB URL:", db_url)
    print("GROQ API KEY:", groq_api_key)
    print("GEMINI API KEY:", gemini_api_key)
    
    qdrant_client = QdrantClient(db_url)
    embedder = SentenceTransformer("BAAI/bge-small-en")
    print("EMBEDDING MODEL LOADED", embedder)
    groq_client = Groq(api_key=groq_api_key)
    print("GROQ CLIENT LOADED")
    gai.configure(api_key=gemini_api_key)
    gemini_client = gai.GenerativeModel('gemini-1.5-flash')
    print("GEMINI CLIENT LOADED")

    collection_name = "python_docs"
    all_retrieved_texts = set()
    query = user_prompt
    pass_count = 0
    final_answer = None
    prev_retrieved_count = 0

    while pass_count < max_passes:
        pass_count += 1
        print(f"\n🔄 **Pass {pass_count}**")

        query_embedding = embedder.encode(query).tolist()
        print("1")
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=20,
            score_threshold=0.7,
            with_payload=True
        )
        print("2")
        retrieved_texts = set()
        retrieved_pages = []
        for hit in search_results:
            if "text" in hit.payload and hit.payload["text"] not in all_retrieved_texts:
                retrieved_texts.add(hit.payload["text"])
                all_retrieved_texts.add(hit.payload["text"])
                page_number = hit.payload.get("page_number")
                if page_number is not None:
                    retrieved_pages.append(page_number)
        print("3")
        for page in retrieved_pages:
            lower_bound = max(0, page - 3)
            upper_bound = page + 3
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
        print(f"\n📚 **Retrieved Texts (Pass {pass_count}):**\n{retrieved_texts_str}")

        summary = " ".join([text[:200] for text in retrieved_texts])
        eval_prompt = f"""You are an AI assistant tasked with answering: '{user_prompt}'.
        Summary of retrieved docs:\n{summary}\n\n
        Full retrieved docs:\n{retrieved_texts_str}\n\n
        Does this provide enough info to answer fully? If yes, provide the answer.
        If no, explain what’s missing and suggest a new query.
        Format:
        - Sufficient: Yes/No (Yes if syntax, examples, edge cases like nested/if-else covered; No if not)
        - Answer: (if Yes, the answer; if No, blank)
        - Missing: (if No, e.g., 'Syntax variants, nested examples, if-else conditions')
        - New Query: (if No, a refined query)"""

        eval_messages = [
            {"role": "system", "content": "You are a precise coding assistant."},
            {"role": "user", "content": eval_prompt}
        ]

        eval_response = None
        try:
            if model == 0:  # Groq
                eval_response = await asyncio.to_thread(
                    groq_client.chat.completions.create,
                    temperature=0.7,
                    n=1,
                    model="deepseek-r1-distill-llama-70b",
                    messages=eval_messages,
                    max_tokens=5000
                )
                eval_response = eval_response.choices[0].message.content
            else:  # Gemini
                eval_response = await asyncio.to_thread(
                    gemini_client.generate_content,
                    eval_prompt  # Gemini takes a string, not messages
                )
                eval_response = eval_response.text
        except Exception as e:
            print(f"Error in eval model call (Pass {pass_count}): {str(e)}")
            raise

        print(f"\n🤓 **Evaluation (Pass {pass_count}):**\n{eval_response}")

        sufficient = "Sufficient: Yes" in eval_response
        if sufficient:
            answer_match = re.search(r"Answer: ([\s\S]+)", eval_response)
            if answer_match:
                final_answer = answer_match.group(1).strip()

        if not sufficient:
            missing_match = re.search(r"Missing: ([\s\S]+?)(?:\nNew Query:|$)", eval_response)
            query_match = re.search(r"New Query: ([\s\S]+)", eval_response)
            if query_match:
                query = query_match.group(1).strip()
                print(f"\n🔍 **Refined Query:** {query}")
            else:
                query = f"Detailed syntax and advanced examples of {user_prompt}"
                print(f"\n🔍 **Default Refined Query:** {query}")

            if pass_count > 1 and len(retrieved_texts) < prev_retrieved_count * 1.2:
                print("\n⚠️ Retrieval gain too low—stopping early")
                break
            prev_retrieved_count = len(retrieved_texts)

    # Final answer generation
    final_prompt = f"""Based on all retrieved Python docs:\n\n{'\n'.join(all_retrieved_texts)}\n\n
    Answer: {user_prompt}"""
    final_messages = [
        {"role": "system", "content": "You are a coding assistant with access to documentation."},
        {"role": "user", "content": final_prompt}
    ]

    final_response = None
    try:
        if model == 0:  # Groq
            final_response = await asyncio.to_thread(
                groq_client.chat.completions.create,
                temperature=1.1,
                n=1,
                model="deepseek-r1-distill-llama-70b",
                messages=final_messages,
                max_tokens=2000
            )
            final_response = final_response.choices[0].message.content
        else:  # Gemini
            final_response = await asyncio.to_thread(
                gemini_client.generate_content,
                final_prompt
            )
            final_response = final_response.text
    except Exception as e:
        print(f"Error in final model call: {str(e)}")
        raise

    final_answer = final_answer or final_response
    print("\n🚀 **Final Answer:**\n")
    print(final_answer)
    return extract_generated_text(final_answer)

# Example usage with FastAPI
app = FastAPI()

# @app.get("/generate")
# async def api_generate(prompt: str, model: int = 0):
#     try:
#         response = await generate_response(prompt, model=model)
#         return {"response": response}
#     except Exception as e:
#         print(f"API Error: {str(e)}")
#         raise

# Standalone test
# async def main():
#     print("Enter your prompt or type 'exit' to quit.")
#     while True:
#         user_prompt = input("")
#         if user_prompt.lower() == "exit":
#             break
#         response = await generate_response(user_prompt, model=0)  # Test with Groq
#         print("\nFinal Output:\n", response)
#         response = await generate_response(user_prompt, model=1)  # Test with Gemini
#         print("\nFinal Output (Gemini):\n", response)

# if __name__ == "__main__":
#     # uvicorn.run(app, host="0.0.0.0", port=8000)
#     # Uncomment to test standalone:
#     asyncio.run(main())