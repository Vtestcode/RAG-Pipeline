import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from RAG_pipeline import build_vector_store
from agent import create_agent

load_dotenv(BASE_DIR / "key.env")

DEFAULT_PDF = r"C:\Users\Visha\Downloads\DE\Vishal.pdf"
pdf_path = os.getenv("RAG_PDF_PATH", DEFAULT_PDF)

st.set_page_config(page_title="RAG Chatbot", page_icon=":speech_balloon:", layout="centered")
st.title("Basic RAG Chatbot")
st.caption(f"Document: `{pdf_path}`")


@st.cache_resource
def get_agent():
    vector_store = build_vector_store(pdf_path)
    return create_agent(vector_store)


agent = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything from your PDF.",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Type your question")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):
            result = agent.invoke({"messages": st.session_state.messages})
            answer = result["messages"][-1].content
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


