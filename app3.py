# Import LangChain dependencies
from langchain.document_loaders import PyPDFLoader, JSONLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Bring in streamlit for UI dev
import streamlit as st
# Bring in watsonx interface
from watsonxlangchain import LangChainInterface

# --- Credentials and LLM Setup ---
# Use Streamlit secrets for secure credentials
# This requires a .streamlit/secrets.toml file with your credentials
try:
    creds = {
        "apikey": st.secrets["watsonx"]["tor1ZjCcjjkupA_XvbLHWvjNuVoy_bDd7iBRogYmtqqI"],
        "url": st.secrets["watsonx"]["https://eu-de.ml.cloud.ibm.com"]
    }
    project_id = st.secrets["watsonx"]["93d14d53-f1e4-4a62-8327-6c2c34f4a413"]
except KeyError:
    st.error("Please configure your `watsonx` secrets in `.streamlit/secrets.toml`.")
    st.stop()

# Create LLM using LangChain
llm = LangChainInterface(
    credentials=creds,
    model="meta-llama/llama-2-70b-chat",
    params={
        "decoding_method": "sample",
        "max_new_tokens": 200,
        "temperature": 0.5,
    },
    project_id=project_id
)

# --- Data Loading and Indexing ---
# This function loads a JSON file and creates a vector store index
@st.cache_resource
def load_json():
    # Update JSON file name here
    json_name = "data.json"
    
    # Load the JSON. Make sure the jq_schema matches your JSON file's structure.
    # The current schema ".text" assumes your JSON has a top-level key named "text".
    try:
        loader = JSONLoader(
            file_path=json_name,
            jq_schema=".text",  # 👈 Change this to match your JSON field
            text_content=False
        )
    except FileNotFoundError:
        st.error(f"Error: The file `{json_name}` was not found.")
        st.stop()

    # Create index (vector database using embeddings)
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders([loader])
    
    return index

# Load the JSON index
index = load_json()

# Create a Q&A chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    input_key="question"
)

# --- Streamlit UI and Chat Logic ---
# Setup the app title
st.title("Rent")

# Setup a session state message variable to hold all the old messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Build a prompt input template to display the prompts
prompt = st.chat_input("Pass your prompt here")

# If the user hits enter then
if prompt:
    # Display the prompt
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Append the new user message into session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Send the prompt to the llm
    with st.spinner("Thinking..."):
        try:
            response = chain.run(prompt)
            # chain.run() should return a string, so no need for an extra check
            reply = response
        except Exception as e:
            reply = f"An error occurred: {e}"
            st.error(reply)

    # Show the LLM response
    with st.chat_message("assistant"):
        st.markdown(reply)

    # Store the LLM response in state
    st.session_state.messages.append({"role": "assistant", "content": reply})
