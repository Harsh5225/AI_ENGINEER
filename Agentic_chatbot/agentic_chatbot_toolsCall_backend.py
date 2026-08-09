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

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

load_dotenv()

# ---------------- LLM ----------------
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# ---------------- Tools ----------------
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

tools = [search_tool, calculator, get_stock_price, get_current_weather]
llm_with_tools = llm.bind_tools(tools)

# ---------------- State ----------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ---------------- Nodes ----------------
def chat_node(state: ChatState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools)

# ---------------- Checkpointer ----------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# ---------------- Graph ----------------
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