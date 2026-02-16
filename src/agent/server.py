"""
RAG agent server: FastAPI app that runs the RAG graph.
Use for API calls or for LangGraph Studio (point Studio at this server's graph).
"""
import os
import sys

from fastapi import FastAPI
from pydantic import BaseModel

# Project root
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.chdir(_project_root)

from src.agent.graph import create_rag_graph, run_rag

app = FastAPI(title="kb_engine_playground RAG Agent", version="0.1.0")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Single-turn RAG: send a question, get an answer."""
    answer = run_rag(req.message)
    return ChatResponse(answer=answer)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def run_server(host: str = "0.0.0.0", port: int = 8123) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8123"))
    run_server(port=port)
