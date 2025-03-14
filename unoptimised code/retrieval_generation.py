from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os
import re

def extract_generated_text(response: str) -> str:
    match = re.search(r"</think>\s*['\"“”]?([\s\S]+?)['\"“”]?$", response.strip())
    print("REGEX MATCHING:", match)
    return match.group(1).strip() if match else response

# Load environment variables
async def generate_response(user_prompt: str) -> str:
    load_dotenv()
    db_url = os.environ.get("QDRANT_DB_URL")
    api_key = os.environ.get("GROQ_API_KEY")

    # Initialize clients
    qdrant_client = QdrantClient(db_url)
    model = SentenceTransformer("BAAI/bge-small-en")
    groq_client = Groq(api_key=api_key)

    # Define collection name
    collection_name = "python_docs"

    # User query
    query = user_prompt
    query_embedding = model.encode(query).tolist()

    # Step 1: Retrieve top 5 relevant chunks
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=5,  # Get top 5 relevant chunks
        with_payload=True  # Retrieve metadata
    )

    # Step 2: Extract `page_number` and primary chunks
    retrieved_pages = []
    retrieved_texts = set()  # Using a set to remove duplicate chunks

    for hit in search_results:
        if "text" in hit.payload:
            retrieved_texts.add(hit.payload["text"])
            page_number = hit.payload.get("page_number")  # Get stored page number
            if page_number is not None:
                retrieved_pages.append(page_number)

    # Step 3: Fetch 3 chunks before and 3 chunks after each retrieved page
    for page in retrieved_pages:
        lower_bound = max(0, page - 3)
        upper_bound = page + 3

        surrounding_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,  # Still using the query embedding
            limit=10,  # Extra limit to ensure relevant context is fetched
            with_payload=True
        )

        for doc in surrounding_results:
            if "text" in doc.payload:
                retrieved_texts.add(doc.payload["text"])

    # Step 4: Convert set to ordered list
    retrieved_texts_list = list(retrieved_texts)

    # Step 5: Construct prompt with retrieved context
    retrieved_texts_str = "\n".join(retrieved_texts_list)

    # print the retrieved texts
    print("\n📚 **Retrieved Texts:**\n")
    print(retrieved_texts_str)

    prompt = f"Based on the given Python documentation, answer the question: {query}"

    system_prompt = f"""You are a coding assistant with access to official documentation. 
    Use the following documentation to answer the user's question accurately. 
    Cite relevant parts when necessary. Provide a detailed and informative response. 
    Whenever possible, include an example to illustrate the explanation. 
    Code snippets or real-world analogies will help make concepts clearer. 
    If you are unsure about a specific detail, it's okay to mention that in your response. 

    These are the relevant documentation chunks for your reference:\n\n{retrieved_texts_str}
    """

    # Step 6: Set up messages for the model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # Step 7: Generate response
    chat_completion = groq_client.chat.completions.create(
        temperature=1.1,
        n=1,
        model="deepseek-r1-distill-llama-70b",
        messages=messages,
        max_tokens=10000
    )

    # Output response
    print("\n🚀 **AI Response:**\n")
    print(chat_completion.choices[0].message.content)

    return extract_generated_text(chat_completion.choices[0].message.content)

# import asyncio

# async def main():
#     print("Enter your prompt or type 'exit' to quit.")
#     user_prompt = input("")
#     while user_prompt != "exit":
#         response = await generate_response(user_prompt)
#         print("\nFinal Output:\n", response)
#         user_prompt = input("")

# asyncio.run(main())
