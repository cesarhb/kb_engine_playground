"""
RAG agent graph: retrieve from pgvector, then generate answer with LLM.
Uses LangChain/LangGraph so it can be run from CLI or LangGraph Studio.
Exposes vector search as a tool (search_kb) for LangGraph Studio / langgraph dev.
"""
import os
import sys
from typing import Annotated, TypedDict

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Re-use ingestion store for vector store and embeddings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.ingestion.store import get_vector_store

# Default collection for tool and graphs
_DEFAULT_COLLECTION = "kb_engine_playground"
_DEFAULT_RETRIEVER_K = 4


def get_llm():
    """Return LLM from env: ollama or openai."""
    provider = (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()
    model = os.getenv("LLM_MODEL") or "llama3.2"

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0)
    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        return ChatOllama(model=model, base_url=base_url, temperature=0)
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def get_retriever(k: int = 4, collection_name: str = "kb_engine_playground"):
    """Build retriever from pgvector store."""
    store = get_vector_store(collection_name=collection_name)
    return store.as_retriever(search_kwargs={"k": k})


def _make_search_kb_tool(collection_name: str = _DEFAULT_COLLECTION, k: int = _DEFAULT_RETRIEVER_K):
    """Build the search_kb tool (vector DB proximity search) for the agent."""
    retriever = get_retriever(k=k, collection_name=collection_name)

    @tool
    def search_kb(query: str) -> str:
        """Search the knowledge base (vector DB) for relevant passages. Use this to find documentation, whitepapers, or repo content before answering."""
        docs = retriever.invoke(query)
        return "\n\n---\n\n".join(d.page_content for d in docs) if docs else "No relevant documents found."

    return search_kb


def _rag_chain(retriever, llm):
    """Classic RAG: format context from retriever, then generate with LLM."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You answer questions based only on the following context. If the context does not contain enough information, say so.\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


class RAGState(TypedDict):
    question: str
    answer: str


def create_rag_graph(collection_name: str = "kb_engine_playground", retriever_k: int = 4):
    """
    Build a simple RAG graph: input question -> retrieve -> generate -> output.
    Returns a compiled LangGraph that can be invoked or served for LangGraph Studio.
    """
    retriever = get_retriever(k=retriever_k, collection_name=collection_name)
    llm = get_llm()
    chain = _rag_chain(retriever, llm)

    def rag_node(state: RAGState) -> dict:
        question = state.get("question") or ""
        answer = chain.invoke(question)
        return {"answer": answer}

    graph = StateGraph(RAGState)
    graph.add_node("rag", rag_node)
    graph.add_edge(START, "rag")
    graph.add_edge("rag", END)

    return graph.compile()


def run_rag(question: str, collection_name: str = "kb_engine_playground") -> str:
    """Run RAG once and return the answer string. Used by CLI."""
    graph = create_rag_graph(collection_name=collection_name)
    result = graph.invoke({"question": question, "answer": ""})
    return result.get("answer") or ""


# --- Agent-with-tool graph for LangGraph Studio / langgraph dev ---

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def create_agent_graph(
    collection_name: str = _DEFAULT_COLLECTION,
    retriever_k: int = _DEFAULT_RETRIEVER_K,
):
    """
    Build an agent graph that has a search_kb tool (vector DB). The LLM can call
    search_kb to retrieve context before answering. Use with LangGraph Studio / langgraph dev.
    """
    search_kb_tool = _make_search_kb_tool(collection_name=collection_name, k=retriever_k)
    llm = get_llm().bind_tools([search_kb_tool])
    tool_node = ToolNode([search_kb_tool])

    system_prompt = (
        "You answer questions using the knowledge base. When you need facts, documentation, or examples, "
        "call the search_kb tool to search the vector DB, then answer based on the retrieved context. "
        "If the context does not contain enough information, say so."
    )

    def agent_node(state: AgentState) -> dict:
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_edge(START, "agent")

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph_builder.add_edge("tools", "agent")

    return graph_builder.compile()


# Entrypoint for langgraph dev / LangGraph Studio (expects "graph" in langgraph.json).
graph = create_agent_graph()
