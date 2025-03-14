from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")

def draft_message(content, role='user'):
    return {
        "content": content,
        "role": role
    }

groq_client = Groq(api_key=api_key)

messages = [
    draft_message(content="You are a chill dude. And you will reply like a chill dude.", role="system")
]

prompt="What do you think of the Python 3.12 documentation?"
messages.append(draft_message(content=prompt))

chat_completion = groq_client.chat.completions.create(
    temperature=1.1,
    n=1,
    model="deepseek-r1-distill-llama-70b",
    messages=messages,
    max_tokens=10000
)

print(chat_completion.choices[0].message.content)