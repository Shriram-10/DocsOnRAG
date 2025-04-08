import requests
import json
from typing import List, Dict

# Define base URLs for the two ports
TUNED_BASE_URL = "http://localhost:8000"  # For tuned-gemini and tuned-gemini-rag
BASE_URL = "http://localhost:8001"        # For gemini and rag

QUERIES = [
    "What is the os module in Python?", "What is sys.argv used for?", "What is a list in Python?",
    "What is the purpose of the random module?", "What is datetime in Python?", "What is a tuple?",
    "What is the math module?", "What is json used for?", "What is the re module?", "What is pathlib?",
    "How do I use os.path.join()?", "How do I use sys.argv in a script?", "How do I use random.randint()?",
    "How do I use datetime.now()?", "How do I use list.append()?", "How do I use tuple unpacking?",
    "How do I use math.sqrt()?", "How do I use json.dumps()?", "How do I use re.match()?",
    "How do I use Path from pathlib?", "Summarize the os module.", "Write an example of a for loop.",
    "Explain the difference between list and tuple.", "Generate a random number between 1 and 10.",
    "How do I read a file in Python?", "What happens if I call sys.exit()?", "Convert a string to uppercase.",
    "Write a simple function to add two numbers.", "How do I handle exceptions in Python?",
    "Compare sort() and sorted()."
]

def fetch_response(base_url: str, endpoint: str, query: str) -> str:
    payload = {"user_prompt": query}
    try:
        response = requests.post(f"{base_url}/{endpoint}", json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            print(f"Error at {endpoint} (port {base_url.split(':')[-1]}) for '{query}': {response.status_code}")
            return f"Error: {response.status_code}"
    except requests.Timeout:
        print(f"Timeout at {endpoint} (port {base_url.split(':')[-1]}) for '{query}'")
        return "Error: Timeout"
    except Exception as e:
        print(f"Exception at {endpoint} (port {base_url.split(':')[-1]}) for '{query}': {str(e)}")
        return f"Error: {str(e)}"

def save_results(results: List[Dict], output_file: str = "output.json"):
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

def collect_responses(output_file: str = "output.json"):
    # Load existing results if file exists
    try:
        with open(output_file, "r") as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        results = []  # Start fresh if file doesn’t exist or is invalid

    for query in QUERIES:
        # Skip if query already processed
        if any(r["query"] == query for r in results):
            print(f"Skipping already processed query: {query}")
            continue
        
        query_result = {
            "query": query,
            "finetuned gemini": fetch_response(TUNED_BASE_URL, "tuned-gemini", query),
            "rag + finetuned gemini": fetch_response(TUNED_BASE_URL, "tuned-gemini-rag", query),
            "gemini": fetch_response(BASE_URL, "gemini", query),
            "rag + gemini": fetch_response(BASE_URL, "rag", query)
        }
        results.append(query_result)
        save_results(results, output_file)  # Save after each query
        print(f"Processed and saved query: {query}")

    return results

def main():
    print("Collecting responses from all endpoints across ports 8000 and 8001...")
    results = collect_responses()

    print("\nSample Results:")
    for i in range(min(3, len(results))):
        print(f"\nQuery: {results[i]['query']}")
        print(f"Finetuned Gemini (8000): {results[i]['finetuned gemini'][:100]}...")
        print(f"RAG + Finetuned Gemini (8000): {results[i]['rag + finetuned gemini'][:100]}...")
        print(f"Gemini (8001): {results[i]['gemini'][:100]}...")
        print(f"RAG + Gemini (8001): {results[i]['rag + gemini'][:100]}...")

if __name__ == "__main__":
    main()