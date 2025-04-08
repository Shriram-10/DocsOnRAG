import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.environ.get("FINETUNED_GEMINI_API_KEY")  # Replace with your Google API key
TUNED_MODEL_ID = "tunedModels/pythonchatbotdataset-6scpyuehcnr5"  # Replace with your fine-tuned model ID

# Initialize the client
genai.configure(api_key=GOOGLE_API_KEY)
tuned_gemini_client = genai.GenerativeModel(model_name=TUNED_MODEL_ID)

# Function to test the model
def test_finetuned_model(query: str) -> str:
    try:
        system_prompt = "You are a precise coding assistant trained to predict the next number or word in a sequence."
        full_prompt = f"{system_prompt}\n\nUser query: {query}"
        response = tuned_gemini_client.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Interactive testing
if __name__ == "__main__":
    print("Testing the fine-tuned Gemini model...")
    while True:
        query = input("Enter a query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        result = test_finetuned_model(query)
        print(f"Response: {result}\n")