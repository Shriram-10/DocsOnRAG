import os
import sys
import spacy
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from datetime import datetime

# Load spaCy + Neo4j
nlp = spacy.load("en_core_web_sm")
load_dotenv()
neo4j_driver = GraphDatabase.driver(os.environ.get("NEO4J_URL", "neo4j://localhost:7687"), auth=("neo4j", "password"))

def heuristic_relations(chunk_text: str, entities: list, doc) -> list:
    print(f"Generating relations for chunk: '{chunk_text[:50]}...'")
    relations = []
    for i in range(len(entities) - 1):
        e1, label1 = entities[i]
        e2, label2 = entities[i + 1]
        print(f"Checking relation between '{e1}' ({label1}) and '{e2}' ({label2})")
        e1_token = next((t for t in doc if t.text == e1.split()[0]), None)
        e2_token = next((t for t in doc if t.text == e2.split()[0]), None)
        if e1_token and e2_token:
            print(f"Tokens found: '{e1_token.text}' (head: {e1_token.head.text}), '{e2_token.text}'")
            if e1_token.head.text == e2_token.text or "contain" in chunk_text.lower():
                relations.append((e1, "CONTAINS", e2))
                print(f"Relation: '{e1}' -[:CONTAINS]-> '{e2}'")
            elif "use" in chunk_text.lower() or (e1_token.dep_ == "nsubj" and e2_token.dep_ in ("dobj", "pobj")):
                relations.append((e1, "USES", e2))
                print(f"Relation: '{e1}' -[:USES]-> '{e2}'")
            elif "equivalent" in chunk_text.lower() or "is" in chunk_text[e1_token.i:e2_token.i].lower():
                relations.append((e1, "EQUIVALENT_TO", e2))
                print(f"Relation: '{e1}' -[:EQUIVALENT_TO]-> '{e2}'")
            else:
                relations.append((e1, "RELATED_TO", e2))
                print(f"Relation: '{e1}' -[:RELATED_TO]-> '{e2}' (fallback)")
        else:
            print(f"No tokens found for '{e1}' or '{e2}' in chunk")
    print(f"Generated {len(relations)} relations for chunk")
    return relations

def create_graph(chunks, driver):
    print(f"Starting graph creation with {len(chunks)} chunks")
    with driver.session() as session:
        for idx, point in enumerate(chunks):
            chunk_text = point.payload["text"]
            chunk_id = point.id
            print(f"Processing chunk {idx + 1}/{len(chunks)} (ID: {chunk_id}): '{chunk_text[:50]}...'")
            
            try:
                doc = nlp(chunk_text)
                entities = [(ent.text, ent.label_) for ent in doc.ents] or [(chunk_text, "CONCEPT")]
                print(f"Extracted {len(entities)} entities: {[(e, l) for e, l in entities]}")
                
                relations = heuristic_relations(chunk_text, entities, doc)
                
                for entity1, relation, entity2 in relations:
                    print(f"Writing relation: '{entity1}' -[:{relation}]-> '{entity2}'")
                    session.run(
                        "MERGE (e1:Entity {name: $name1}) "
                        "MERGE (e2:Entity {name: $name2}) "
                        f"MERGE (e1)-[:{relation}]->(e2)",
                        name1=entity1, name2=entity2
                    )
                
                for entity, label in entities:
                    print(f"Writing entity: '{entity}' (type: {label}) for chunk {chunk_id}")
                    session.run(
                        "MERGE (e:Entity {name: $name, type: $type}) "
                        "MERGE (c:Chunk {id: $chunk_id, text: $text}) "
                        "MERGE (c)-[:CONTAINS]->(e)",
                        name=entity, type=label, chunk_id=chunk_id, text=chunk_text
                    )
                print(f"Completed chunk {idx + 1}/{len(chunks)} (ID: {chunk_id})")
            except Exception as e:
                print(f"Error processing chunk {chunk_id}: {str(e)}")

    print(f"Graph creation completed with ~{len(chunks)} chunks, ~{len(chunks) * 3}–{len(chunks) * 5} nodes")

# Retrieve and build (~143k local)
print("Starting chunk retrieval from Qdrant")
qdrant_client = QdrantClient(os.environ.get("QDRANT_DB_URL"))
chunks = []
offset = None
batch_size = 1000
while True:
    points, next_offset = qdrant_client.scroll(
        collection_name="python_docs",
        limit=batch_size,
        with_payload=True,
        with_vectors=True,
        offset=offset
    )
    chunks.extend(points)
    print(f"Retrieved {len(points)} chunks, total so far: {len(chunks)}")
    if next_offset is None:
        break
    offset = next_offset

print(f"Retrieved {len(chunks)} chunks total from Qdrant")
create_graph(chunks, neo4j_driver)
print(f"Graph created with ~{len(chunks)} chunks—check graph_creation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log for dope details!")