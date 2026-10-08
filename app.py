import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv
from google.genai import errors
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()

DATA_DIR = "data"
INDEX_DIR = "storage"
MODEL_NAME = "gemini-3.6-flash"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

CLIENT_ERROR_HINTS = {
    401: (
        "Google rejected the API key. Check GEMINI_API_KEY in your .env "
        "file, then restart the app."
    ),
    403: (
        "Google refused this request. Check that GEMINI_API_KEY in your "
        ".env file is valid, then restart the app."
    ),
    404: (
        f"The model '{MODEL_NAME}' is not available. Change MODEL_NAME at "
        "the top of app.py to a current Gemini model."
    ),
    429: "Google's rate limit was reached. Wait a minute, then ask again.",
}


def get_api_key():
    """Return the Gemini API key, or stop the app if it is missing."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        st.error(
            "GEMINI_API_KEY was not found. Create a file named .env in the "
            "project folder, next to app.py, with the line "
            "GEMINI_API_KEY=your-key-here. Save it, then restart the app."
        )
        st.stop()
    return api_key


def validate_data_dir():
    """Stop the app if the data folder is missing or is not a folder."""
    data_path = Path(DATA_DIR)
    if not data_path.is_dir():
        st.error(
            f"The app expected a folder named '{DATA_DIR}' at "
            f"{data_path.resolve()}, but it was not found or is not a "
            f"folder. Create a folder named '{DATA_DIR}' next to app.py, "
            "put the handbook inside it, then restart the app."
        )
        st.stop()


def validate_data_files():
    """Stop the app if the data folder has no files to index.

    Hidden files such as .DS_Store are ignored, so a folder that only
    holds hidden files counts as empty.
    """
    files = [
        item
        for item in Path(DATA_DIR).iterdir()
        if item.is_file() and not item.name.startswith(".")
    ]
    if not files:
        st.error(
            f"The '{DATA_DIR}' folder is empty, so there is nothing to "
            "search. Add the student handbook file to it, then restart "
            "the app."
        )
        st.stop()


def load_or_build_index():
    """Load the saved index from disk, or build it and save it."""
    index_path = Path(INDEX_DIR)
    if index_path.is_dir() and any(index_path.iterdir()):
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage_context)
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    return index


@st.cache_resource(show_spinner="Indexing the handbook...")
def get_query_engine(api_key):
    """Set up the models once and return a cached query engine."""
    Settings.llm = GoogleGenAI(model=MODEL_NAME, api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    index = load_or_build_index()
    return index.as_query_engine()


def ask_question(query_engine, prompt):
    """Return the answer to one question, or None if the request fails."""
    try:
        with st.spinner("Searching..."):
            response = query_engine.query(prompt)
        return response.response or (
            "No answer came back. Try rephrasing your question."
        )
    except errors.ClientError as e:
        hint = CLIENT_ERROR_HINTS.get(e.code, "Please try again.")
        st.error(
            f"Google could not answer this question (error {e.code}). "
            f"{hint} The app is still running, so you can ask again."
        )
    except errors.ServerError:
        st.error(
            "Google's servers had a temporary problem. Wait a moment, "
            "then ask again."
        )
    except (httpx.TransportError, ConnectionError, TimeoutError):
        st.error(
            "The app could not reach Google. Check your internet "
            "connection, then ask again."
        )
    except Exception as e:
        st.error(f"Something unexpected went wrong: {e}. Please ask again.")
    return None


st.title("Babson Student Handbook Chatbot")

api_key = get_api_key()
validate_data_dir()
validate_data_files()

try:
    query_engine = get_query_engine(api_key)
except (OSError, ValueError) as e:
    st.error(
        f"The handbook could not be indexed: {e}. Check that the files in "
        f"'{DATA_DIR}' are valid and that you were online the first time "
        f"the embedding model downloaded. If a '{INDEX_DIR}' folder "
        "exists, delete it. Then restart the app."
    )
    st.stop()
except Exception as e:
    st.error(
        f"The app could not build its query engine: {e}. Restart the app. "
        f"If this keeps happening, delete the '{INDEX_DIR}' folder."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the handbook"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = ask_question(query_engine, prompt)
        if answer is not None:
            st.markdown(answer)

    if answer is not None:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
