from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import the RAG function
from optimisations.active_retrieval import generate_response  # Update with the actual module name
app = FastAPI()

# Request model
class QueryRequest(BaseModel):
    user_prompt: str

@app.post("/generate_response")
async def generate(query: QueryRequest):
    try:
        response = await generate_response(query.user_prompt, max_passes=3, model = 1)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))