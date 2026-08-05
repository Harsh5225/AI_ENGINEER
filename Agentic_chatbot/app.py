import streamlit as st
from langchain_core.messages import HumanMessage
from agentic_chatbot_backend import chatbot   # <-- this one import brings in the entire "brain"

st.title("My Agentic Chatbot")

# a fixed conversation ID — same thread for the whole app session
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# ---- Step A: set up chat history storage (survives Streamlit reruns) ----
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# ---- Step B: redraw all past messages every time the page reruns ----
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# ---- Step C: get new input from the user ----
user_input = st.chat_input('Type here')

if user_input:
    # show the user's message immediately + save it
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # ---- Step D: send it to the backend ----
    response = chatbot.invoke(
        {'messages': [HumanMessage(content=user_input)]},
        config=CONFIG
    )
    ai_message = response['messages'][-1].content

    # show the AI's reply + save it
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)