import sqlite3
import math
import os
from typing import Any, TypedDict, Annotated

import requests
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import streamlit as st

load_dotenv()

# ---------------- LLM (Groq) ----------------
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# ---------------- Embeddings (free, local) ----------------
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings_model()


def ingest_rag_document(file_path):
    DB_PATH = "faiss_db"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)


def get_retriever():
    DB_PATH = "faiss_db"
    vector_store = FAISS.load_local(
        folder_path=DB_PATH, embeddings=embeddings, allow_dangerous_deserialization=True
    )
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


@tool
def rag_tool(query: str) -> str:
    """Retrieve relevant information from the PDF document."""
    retriever = get_retriever()
    documents = retriever.invoke(query)
    if not documents:
        return "No relevant information was found in the PDF."
    formatted = []
    for i, d in enumerate(documents, start=1):
        formatted.append(
            f"Document {i}\nSource: {d.metadata.get('source', 'Unknown')}\n"
            f"Page: {d.metadata.get('page', 'Unknown')}\nContent: {d.page_content}"
        )
    return "\n\n".join(formatted)


search_tool = TavilySearch(max_results=5, topic="general", search_depth="advanced")

@tool
def calculator(expression: str) -> str:
    """Useful for simple math calculations. Example: 2 + 2, math.sqrt(16), 10 * 5"""
    try:
        allowed = {"math": math, "abs": abs, "round": round, "min": min, "max": max, "sum": sum}
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url, timeout=10)
    return r.json()

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.
    HUMAN-IN-THE-LOOP: pauses and waits for human approval ("yes"/"no").
    """
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")
    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol, "quantity": quantity,
        }
    return {
        "status": "cancelled",
        "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
        "symbol": symbol, "quantity": quantity,
    }

@tool
def get_current_weather(location: str) -> str:
    """Get the current real-time weather for a given city or location."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather API key is missing. Set the OPENWEATHER_API_KEY environment variable."
    try:
        geo = requests.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={"q": location, "limit": 1, "appid": api_key}, timeout=10,
        )
        geo.raise_for_status()
        locations: list[dict[str, Any]] = geo.json()
        if not locations:
            return f"Could not find the location: {location}"
        lat, lon = locations[0]["lat"], locations[0]["lon"]
        resolved_name, country = locations[0].get("name", location), locations[0].get("country", "")
        w = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}, timeout=10,
        )
        w.raise_for_status()
        data = w.json()
        return (
            f"Current weather in {resolved_name}, {country}: "
            f"{data['weather'][0]['description'].title()}, {data['main']['temp']}°C"
        )
    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response is not None else "unknown"
        if status_code == 401:
            return "The OpenWeather API key is invalid or inactive."
        return f"Weather API returned an HTTP error: {status_code}"
    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"


tools = [search_tool, calculator, get_stock_price, get_current_weather, rag_tool, purchase_stock]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"
            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded PDF or document.\n"
            "- Use `search_tool` for current events or information needing internet search.\n"
            "- Use `calculator` for mathematical calculations.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
            "- Use `purchase_stock` when the user wants to purchase a stock.\n"
            "- Use `get_current_weather` when the user asks about current weather.\n\n"
            "Answer general questions directly when no tool is required. "
            "Do not invent information from the uploaded document."
        )
    )
    response = llm_with_tools.invoke([system_message, *state["messages"]])
    return {"messages": [response]}


tool_node = ToolNode(tools)

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpoint)


def retrieve_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config["configurable"]["thread_id"])
    return list(all_threads)