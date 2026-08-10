import os
import uuid

import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command

from agentic_chatbot_hitl import (
    chatbot,
    retrieve_all_threads,
    ingest_rag_document,
)



# Helper functions — threading

def generate_thread_id():
    """Generate a unique ID for a new conversation."""
    return str(uuid.uuid4())


def add_thread(thread_id):
    """Register a thread ID in the sidebar list (no duplicates)."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    """Start a brand-new conversation."""
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    st.session_state["pending_approval"] = None
    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id):
    """Pull a thread's saved messages from the LangGraph checkpointer."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def get_thread_title(thread_id):
    """
    Derive a readable sidebar label from the thread's first user message,
    cached in session_state so we don't re-read the checkpointer every rerun.
    """
    if "thread_titles" not in st.session_state:
        st.session_state["thread_titles"] = {}

    if thread_id in st.session_state["thread_titles"]:
        return st.session_state["thread_titles"][thread_id]

    title = "New Conversation"
    messages = load_conversation(thread_id)
    for message in messages:
        if isinstance(message, HumanMessage) and message.content:
            title = message.content[:40] + ("..." if len(message.content) > 40 else "")
            break

    st.session_state["thread_titles"][thread_id] = title
    return title


def delete_thread(thread_id):
    """Remove a thread from the sidebar (and try to delete it from the DB)."""
    if thread_id in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].remove(thread_id)
    st.session_state["thread_titles"].pop(thread_id, None)

    try:
        chatbot.checkpointer.delete_thread(thread_id)
    except Exception:
        pass

    if st.session_state["thread_id"] == thread_id:
        reset_chat()


# ============================================================
# Helper functions — RAG document management
# ============================================================

UPLOAD_DIR = "uploaded_docs"
FAISS_DB_PATH = "faiss_db"


def save_uploaded_file(uploaded_file):
    """Save a Streamlit-uploaded file to disk so PyPDFLoader can read it."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def is_document_ready():
    """Check whether a FAISS index already exists on disk."""
    return os.path.isdir(FAISS_DB_PATH) and len(os.listdir(FAISS_DB_PATH)) > 0



# App setup

st.title("Agentic Chatbot with LangGraph")

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

if "active_document_name" not in st.session_state:
    st.session_state["active_document_name"] = None

if "pending_approval" not in st.session_state:
    st.session_state["pending_approval"] = None

add_thread(st.session_state["thread_id"])



# Sidebar — Document knowledge base status

st.sidebar.title("📄 Document Knowledge Base")

if is_document_ready():
    doc_label = st.session_state.get("active_document_name") or "a document"
    st.sidebar.success(f"✅ Knowledge base ready ({doc_label})")
else:
    st.sidebar.info("ℹ️ No document uploaded yet — attach a PDF using the + button below the chat.")

st.sidebar.divider()


# Sidebar — conversation list

st.sidebar.title("My Conversations")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

search_query = st.sidebar.text_input(
    "🔍 Search", "", label_visibility="collapsed", placeholder="Search conversations"
)

st.sidebar.divider()

# Newest conversation first
ordered_threads = st.session_state["chat_threads"][::-1]

for thread_id in ordered_threads:
    title = get_thread_title(thread_id)

    if search_query and search_query.lower() not in title.lower():
        continue

    is_active = (thread_id == st.session_state["thread_id"])
    label = f"💬 {title}"

    col1, col2 = st.sidebar.columns([5, 1])

    with col1:
        button_type = "primary" if is_active else "secondary"
        if st.button(label, key=f"select-{thread_id}", use_container_width=True, type=button_type):
            st.session_state["thread_id"] = thread_id
            st.session_state["pending_approval"] = None
            messages = load_conversation(thread_id)

            temp_messages = []
            for message in messages:
                if isinstance(message, HumanMessage):
                    role = "user"
                elif isinstance(message, AIMessage):
                    role = "assistant"
                else:
                    continue
                temp_messages.append({"role": role, "content": message.content})

            st.session_state["message_history"] = temp_messages
            st.rerun()

    with col2:
        if st.button("🗑️", key=f"delete-{thread_id}"):
            delete_thread(thread_id)
            st.rerun()


# ============================================================
# Main chat interface — display history
# ============================================================

for message in st.session_state["message_history"]:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.text(message["content"])


# Chat input — supports text AND inline PDF attachment via "+"

prompt = st.chat_input(
    "Type here, or attach a PDF using +",
    accept_file=True,
    file_type=["pdf"],
)

if prompt:
    user_input = prompt.text
    uploaded_files = prompt.files  # list, possibly empty

    # ---- Handle any attached PDF(s) first ----
    if uploaded_files:
        for f in uploaded_files:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(f"📎 Uploaded: **{f.name}**")

            with st.status(f"Processing {f.name}...", expanded=True) as status:
                st.write("Saving file...")
                file_path = save_uploaded_file(f)

                st.write("Splitting into chunks and generating embeddings...")
                ingest_rag_document(file_path)

                st.session_state["active_document_name"] = f.name
                status.update(label=f"{f.name} ready!", state="complete", expanded=False)

        st.rerun()

    # ---- Handle a normal text message ----
    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.text(user_input)

        CONFIG = {
            "configurable": {"thread_id": st.session_state["thread_id"]},
            "metadata": {"thread_id": st.session_state["thread_id"]},
            "run_name": "Chat_stream",
        }

        with st.chat_message("assistant", avatar="🤖"):
            tool_status = st.empty()
            tools_used = []

            def ai_only_stream():
                for message_chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    if isinstance(message_chunk, AIMessage) and message_chunk.tool_calls:
                        for tc in message_chunk.tool_calls:
                            tool_name = tc.get("name", "tool")
                            if tool_name not in tools_used:
                                tools_used.append(tool_name)
                            tool_status.markdown(f"🔧 *Calling `{tool_name}`...*")

                    elif isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        tool_status.markdown(f"✅ *`{tool_name}` done*")

                    elif isinstance(message_chunk, AIMessage) and message_chunk.content:
                        yield message_chunk.content

            with st.spinner("Thinking..."):
                ai_message = st.write_stream(ai_only_stream())

            if tools_used:
                tool_status.caption(f"🛠️ Used: {', '.join(tools_used)}")
            else:
                tool_status.empty()

        # Check if the graph paused waiting for human approval (e.g. purchase_stock)
        state_after = chatbot.get_state(CONFIG)
        pending_interrupts = []
        for task in state_after.tasks:
            pending_interrupts.extend(getattr(task, "interrupts", []) or [])

        if pending_interrupts:
            st.session_state["pending_approval"] = pending_interrupts[0].value
        else:
            st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

            if st.session_state["thread_id"] not in st.session_state["thread_titles"] or \
               st.session_state["thread_titles"].get(st.session_state["thread_id"]) == "New Conversation":
                title = user_input[:40] + ("..." if len(user_input) > 40 else "")
                st.session_state["thread_titles"][st.session_state["thread_id"]] = title


# Human-in-the-loop approval UI

if st.session_state.get("pending_approval"):
    st.warning(f"⚠️ {st.session_state['pending_approval']}")
    col1, col2 = st.columns(2)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
    }

    with col1:
        if st.button("✅ Approve", use_container_width=True):
            result = chatbot.invoke(Command(resume="yes"), config=CONFIG)
            ai_message = result["messages"][-1].content
            st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
            st.session_state["pending_approval"] = None
            st.rerun()

    with col2:
        if st.button("❌ Reject", use_container_width=True):
            result = chatbot.invoke(Command(resume="no"), config=CONFIG)
            ai_message = result["messages"][-1].content
            st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
            st.session_state["pending_approval"] = None
            st.rerun()