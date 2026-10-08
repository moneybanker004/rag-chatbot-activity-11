import os

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()

DATA_DIR = "data"

Settings.llm = GoogleGenAI(
    model="gemini-3.6-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


@st.cache_resource(show_spinner="Indexing the handbook...")
def get_query_engine():
    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY not found. Add it to your .env file and restart.")
        st.stop()
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()


st.title("Undergraduate Student Handbook Chatbot")

query_engine = get_query_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the handbook"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = query_engine.query(prompt)
        answer = response.response
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})