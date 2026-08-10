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

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
# rag
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
load_dotenv()

#LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
# embeddings
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
        folder_path=DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document.
    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
    """
    retriever = get_retriever()
    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the PDF."

    formatted_documents = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")
        formatted_documents.append(
            f"Document {index}\nSource: {source}\nPage: {page}\nContent: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)


# Tools 
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
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey={api_key}"
    )
    r = requests.get(url, timeout=10)
    return r.json()

@tool
def get_current_weather(location: str) -> str:
    """Get the current real-time weather for a given city or location."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather API key is missing. Set the OPENWEATHER_API_KEY environment variable."
    try:
        geocoding_url = "https://api.openweathermap.org/geo/1.0/direct"
        geo_response = requests.get(geocoding_url, params={"q": location, "limit": 1, "appid": api_key}, timeout=10)
        geo_response.raise_for_status()
        locations: list[dict[str, Any]] = geo_response.json()
        if not locations:
            return f"Could not find the location: {location}"

        latitude, longitude = locations[0]["lat"], locations[0]["lon"]
        resolved_name = locations[0].get("name", location)
        country = locations[0].get("country", "")

        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        weather_response = requests.get(
            weather_url,
            params={"lat": latitude, "lon": longitude, "appid": api_key, "units": "metric"},
            timeout=10,
        )
        weather_response.raise_for_status()
        data = weather_response.json()
        return f"Current weather in {resolved_name}, {country}: {data['weather'][0]['description'].title()}, {data['main']['temp']}°C"

    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response is not None else "unknown"
        if status_code == 401:
            return "The OpenWeather API key is invalid or inactive."
        return f"Weather API returned an HTTP error: {status_code}"
    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"

tools = [
    search_tool,
    calculator,
    get_stock_price,
    get_current_weather,
    rag_tool
]

llm_with_tools = llm.bind_tools(tools)

#  State 
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

#  Nodes 
def chat_node(state: ChatState):
    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"
            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded PDF or document. "
            "Always retrieve relevant document content before answering PDF-related questions.\n"
            "- Use `search_tool` for current events, recent information, or information "
            "that requires an internet search.\n"
            "- Use `calculator` for mathematical calculations. Do not calculate complex "
            "expressions manually when the calculator is available.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
            "- Use `get_current_weather` when the user asks about current weather for a location.\n\n"
            "Answer general questions directly when no tool is required. "
            "Do not invent information from the uploaded document. "
            "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
            "After receiving a tool result, provide a clear and helpful final answer."
        )
    )
    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

#  Checkpointer 
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

#  Graph 
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)