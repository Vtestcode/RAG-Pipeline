import sys
import os
from pathlib import Path

# Ensure the script's own directory is on sys.path so sibling modules are found
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / "key.env")

from fastapi import FastAPI
from RAG_pipeline import build_vector_store
from agent import create_agent


app = FastAPI()

pdf_path = os.getenv("RAG_PDF_PATH", r"C:\Users\Visha\Downloads\DE\Vishal.pdf")

vector_store = build_vector_store(pdf_path)
agent = create_agent(vector_store)


@app.post("/ask")
async def ask_question(query: str):
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    response = result["messages"][-1].content
    return {"response": response}
