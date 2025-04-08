from google import genai

client = genai.Client() # Get the key from the GOOGLE_API_KEY env variable

for model_info in client.models.list():
    print(model_info.name)