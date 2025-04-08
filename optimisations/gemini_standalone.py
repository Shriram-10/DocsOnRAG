import os
import google.generativeai as gai
from dotenv import load_dotenv

# Load env vars
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

# Configure Gemini
gai.configure(api_key=GEMINI_API_KEY)
gemini_client = gai.GenerativeModel('gemini-1.5-flash')

# Test queries
queries = [
    "What is the os module in Python?",
    "How do I use sys.argv?",
    "Write a for loop example."
]

# Run
for query in queries:
    try:
        system_prompt = "You are a precise coding assistant."
        full_prompt = f"{system_prompt}\n\nUser query: {query}"
        response = gemini_client.generate_content(full_prompt)
        print(f"\nQuery: {query}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error for '{query}': {str(e)}")

print("Done!")