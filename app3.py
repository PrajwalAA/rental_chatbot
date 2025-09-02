# app.py

# --- Imports ---
import os
import streamlit as st
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub


# --- HuggingFace API Key Setup ---
st.sidebar.title("🔑 API Configuration")

# Option 1: Try environment variable first
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Option 2: Let user enter manually (stored in session_state)
if "hf_token" not in st.session_state:
    st.session_state.hf_token = hf_token if hf_token else ""

st.session_state.hf_token = st.sidebar.text_input(
    "Enter Hugging Face API Token:",
    value=st.session_state.hf_token,
    type="password",
    placeholder="paste your API key here"
)

if not st.session_state.hf_token:
    st.error("⚠️ Please provide a Hugging Face API token to continue.")
    st.stop()


# --- HuggingFace LLM ---
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    huggingfacehub_api_token=st.session_state.hf_token,
    model_kwargs={"temperature": 0.5, "max_length": 200}
)


# --- Data Loading and Indexing ---
@st.cache_resource
def load_json_index():
    json_name = "data.json"  # 👈 your dataset

    try:
        loader = JSONLoader(
            file_path=json_name,
            jq_schema=".text",   # adjust based on your JSON structure
            text_content=False
        )
    except FileNotFoundError:
        st.error(f"Error: The file `{json_name}` was not found.")
        st.stop()

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = loader.load_and_split(text_splitter=splitter)

    # Build embeddings + FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore


# Load JSON vector index
vectorstore = load_json_index()

# Build Retrieval QA
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)


# --- Streamlit UI ---
st.title("🏠 Rental Prediction Chatbot")

# Maintain chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Ask me about rentals...")

if prompt:
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Process with LLM
    with st.spinner("Thinking..."):
        try:
            response = chain.run(prompt)
        except Exception as e:
            response = f"⚠️ Error: {e}"

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
