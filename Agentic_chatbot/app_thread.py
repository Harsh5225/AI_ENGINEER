from agentic_chatbot_toolsCall_backend import chatbot, retrieve_all_threads
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
import streamlit as st
import uuid

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

    # Best-effort DB cleanup — not all checkpointer versions support this
    try:
        chatbot.checkpointer.delete_thread(thread_id)
    except Exception:
        pass

    if st.session_state["thread_id"] == thread_id:
        reset_chat()

st.title("Agentic Chatbot with LangGraph")

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

add_thread(st.session_state["thread_id"])

# Sidebar — conversation list

st.sidebar.title("My Conversations")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

search_query = st.sidebar.text_input("🔍 Search", "", label_visibility="collapsed", placeholder="Search conversations")

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



# Main chat interface

for message in st.session_state["message_history"]:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.text(message["content"])

user_input = st.chat_input("Type here")

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
        tool_status = st.empty()   # single placeholder, updated in place — no stray rows
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

        # once the answer is fully streamed, clear the tool-status line
        # (or replace with a small permanent caption if you want a record of what ran)
        if tools_used:
            tool_status.caption(f"🛠️ Used: {', '.join(tools_used)}")
        else:
            tool_status.empty()


    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

    # New conversation's title wasn't known until just now — cache it
    if st.session_state["thread_id"] not in st.session_state["thread_titles"] or \
       st.session_state["thread_titles"].get(st.session_state["thread_id"]) == "New Conversation":
        title = user_input[:40] + ("..." if len(user_input) > 40 else "")
        st.session_state["thread_titles"][st.session_state["thread_id"]] = title