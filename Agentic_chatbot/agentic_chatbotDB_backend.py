from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
load_dotenv()

# 1. Set up the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# 2. Define the shared state — what data flows through the graph
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  

# 3. Define the node 
def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

# 4. DB 
conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)


# 5. Build the graph
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# 6. Compile — this is the final object the frontend will use
chatbot = graph.compile(checkpointer=checkpointer)


#7
# PROBLEM: st.session_state["chat_threads"] still resets to [] on every app restart (Streamlit's own memory, separate from the DB) → sidebar looks empty even though conversations are saved.
''' Fix — pull thread list from the DB itself'''
def retrieve_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
