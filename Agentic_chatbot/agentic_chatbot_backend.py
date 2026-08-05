from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

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

# 4. Set up memory (so the chatbot remembers past turns)
checkpointer = MemorySaver()

# 5. Build the graph
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# 6. Compile — this is the final object the frontend will use
chatbot = graph.compile(checkpointer=checkpointer)