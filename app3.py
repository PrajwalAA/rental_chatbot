# Import Langchain dependencies
from langchain.document_loaders import JSONLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Bring in streamlit for UI dev
import streamlit as st
# Bring in watsonx interface
from watsonxlangchain import LangChainInterface
from typing import Dict

# --- UI for user inputs in the sidebar ---
st.sidebar.title("Configuration")

# Use st.secrets to suggest a best practice for handling secrets
# The user will enter their key in the text input field
api_key = st.sidebar.text_input(
    "IBM watsonx API Key",
    type="password",
    help="Enter your IBM Cloud API Key."
)

project_id = st.sidebar.text_input(
    "IBM watsonx Project ID",
    help="Enter your IBM watsonx project ID."
)

model_choices = {
    "Llama-2-70b-chat": "meta-llama/llama-2-70b-chat",
    "Flan-T5-xxl": "google/flan-t5-xxl",
    "Granite-20b-instruct": "ibm/granite-20b-instruct-v1"
}

selected_model_name = st.sidebar.selectbox(
    "Select LLM Model",
    options=list(model_choices.keys())
)
selected_model = model_choices[selected_model_name]

# Placeholder for JSON file and JQ schema input
# In a real app, you would use st.file_uploader, but for this example,
# we'll assume a hardcoded file for simplicity and to match the original code structure.
json_name = st.sidebar.text_input(
    "JSON File Name",
    value="data.json",
    disabled=True,
    help="This example uses a hardcoded file name for demonstration. "
         "In a real application, you would use st.file_uploader."
)

jq_schema = st.sidebar.text_input(
    "JQ Schema",
    value=".text",
    help="Enter the JQ schema to parse your JSON data."
)

# Function to safely load the JSON and create the index
@st.cache_resource(experimental_allow_widgets=True)
def load_and_create_index(api_key: str, project_id: str, json_file: str, jq_schema: str):
    """Loads a JSON file and creates a vector store index."""
    if not api_key or not project_id:
        st.error("Please provide your API Key and Project ID in the sidebar.")
        return None, None

    # Setup credentials dictionary using the user-provided API key
    creds = {
        "apikey": api_key,
        "url": "https://eu-de.ml.cloud.ibm.com"
    }

    # Create LLM using LangChain
    llm = LangChainInterface(
        credentials=creds,
        model=selected_model,
        params={
            "decoding_method": "sample",
            "max_new_tokens": 200,
            "temperature": 0.5,
        },
        project_id=project_id
    )

    try:
        loader = JSONLoader(
            file_path=json_file,
            jq_schema=jq_schema,
            text_content=False
        )

        # Create index (vector database using embeddings)
        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        ).from_loaders([loader])

        return index, llm
    except Exception as e:
        st.error(f"Error loading data or creating index: {e}")
        return None, None

# --- Main App Logic ---
st.title("Rent Analysis Bot")
st.write("A Q&A bot for your JSON data. "
         "Please configure your settings in the sidebar.")

# Load the index and LLM only if the necessary credentials are provided
if api_key and project_id:
    index, llm = load_and_create_index(api_key, project_id, json_name, jq_schema)

    if index and llm:
        # Create a Q&A chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=index.vectorstore.as_retriever(),
            input_key="question"
        )

        # Setup a session state message variable to hold all the old messages
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display all the historical messages
        for message in st.session_state.messages:
            st.chat_message(message["role"]).markdown(message["content"])

        # Build a prompt input template to display the prompts
        prompt = st.chat_input("Pass your prompt here")

        # If the user hits enter then
        if prompt:
            # Display the prompt
            st.chat_message("user").markdown(prompt)
            # Append the new user message into session state
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Thinking..."):
                # Send the prompt to the llm
                response = chain.run(prompt)

                # Extract text safely (if response is a LangChain object)
                reply = response if isinstance(response, str) else response.text

            # Show the LLM response
            st.chat_message("assistant").markdown(reply)

            # Store the LLM response in state
            st.session_state.messages.append({"role": "assistant", "content": reply})
else:
    st.info("Please fill out the API Key and Project ID in the sidebar to begin.")
