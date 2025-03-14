import os
import re
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from groq import Groq
from dotenv import load_dotenv
from graph_construction import retrieve_all_chunks, create_graph

import os
import sys
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def extract_generated_text(response: str) -> str:
    match = re.search(r"(?:</think>\s*|Answer:\s*)([\s\S]*)", response.strip())
    return match.group(1).strip() if match else response.strip()

async def generate_response(user_prompt: str, max_passes: int = 3) -> str:
    load_dotenv()
    qdrant_client = QdrantClient(os.environ.get("QDRANT_DB_URL"))
    model = SentenceTransformer("BAAI/bge-small-en")
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    neo4j_driver = GraphDatabase.driver(os.environ.get("NEO4J_URI", "neo4j://localhost:7687"), auth=("neo4j", "password"))

    # Pre-build graph (~143k local, ~2–3M prod)
    chunks = qdrant_client.scroll(collection_name="python_docs", limit=1000, with_payload=True, with_vectors=True)[0]
    create_graph(chunks, neo4j_driver)

    collection_name = "dev_docs"
    all_retrieved_texts = set()
    query = user_prompt
    pass_count = 0
    final_answer = None

    while pass_count < max_passes:
        pass_count += 1
        print(f"\n🔄 **Pass {pass_count}**")

        query_embedding = model.encode(query).tolist()
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=20,
            score_threshold=0.7,
            with_payload=True
        )

        with neo4j_driver.session() as session:
            graph_results = session.run(
                "MATCH (e1:Entity)-[:RELATED_TO|:GENERATES|:EQUIVALENT_TO|:USES|:CONTAINS*1..2]->(e2:Entity)-[:CONTAINS]-(c:Chunk) "
                "WHERE toLower(e1.name) CONTAINS toLower($query) "
                "RETURN c.text AS text LIMIT 20",
                query=query
            )
            graph_texts = {record["text"] for record in graph_results}

        retrieved_texts = {hit.payload["text"] for hit in search_results if hit.payload["text"] not in all_retrieved_texts}
        retrieved_texts.update(graph_texts - all_retrieved_texts)
        all_retrieved_texts.update(retrieved_texts)

        retrieved_texts_str = "\n".join(retrieved_texts)
        print(f"\n📚 **Retrieved Texts (Pass {pass_count}):**\n{retrieved_texts_str}")

        eval_prompt = f"""You are an AI assistant answering: '{user_prompt}'.
        System-provided docs:\n\n{retrieved_texts_str}\n\n
        Sufficient? Consider advanced cases.
        Format:
        - Sufficient: Yes/No
        - Answer: (if Yes, the answer; if No, blank)
        - Missing: (if No, e.g., 'Syntax variants')
        - New Query: (if No, refined query)"""

        eval_response = groq_client.chat.completions.create(
            temperature=0.7,
            n=1,
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "system", "content": "Precise coding assistant."}, {"role": "user", "content": eval_prompt}],
            max_tokens=1000
        ).choices[0].message.content

        print(f"\n🤓 **Evaluation (Pass {pass_count}):**\n{eval_response}")

        sufficient = "Sufficient: Yes" in eval_response
        if sufficient:
            answer_match = re.search(r"Answer: ([\s\S]+)", eval_response)
            if answer_match:
                final_answer = answer_match.group(1).strip()
                break

        query_match = re.search(r"New Query: ([\s\S]+)", eval_response)
        query = query_match.group(1).strip() if query_match else f"Detailed syntax and advanced examples of {user_prompt}"
        print(f"\n🔍 **Refined Query:** {query}")

    final_response = groq_client.chat.completions.create(
        temperature=1.1,
        n=1,
        model="deepseek-r1-distill-llama-70b",
        messages=[{"role": "system", "content": "Coding assistant."}, {"role": "user", "content": f"All docs:\n\n{'\n'.join(all_retrieved_texts)}\n\nAnswer: {user_prompt}"}],
        max_tokens=2000
    ).choices[0].message.content

    final_answer = final_answer or final_response
    print("\n🚀 **Final Answer:**\n")
    print(final_answer)
    return extract_generated_text(final_answer)