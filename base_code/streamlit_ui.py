import streamlit as st
import requests

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/generate_response"

# Set up page
st.set_page_config(page_title="AI Coding Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AI Coding Assistant")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask a question about Python...")

if user_input:
    # Add user message to chat
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Display loading state
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ Generating response...")

    try:
        # Call FastAPI backend
        response = requests.post(API_URL, json={"user_prompt": user_input})
        response_data = response.json()

        if "response" in response_data:
            ai_response = response_data["response"]
        else:
            ai_response = "⚠️ Something went wrong. Please try again."

    except Exception as e:
        ai_response = f"❌ Error: {str(e)}"

    # Update chat history with AI response
    st.session_state["messages"].append({"role": "assistant", "content": ai_response})

    # Replace loading message with actual response
    placeholder.markdown(ai_response)
