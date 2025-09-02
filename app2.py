# Import Langchain dependencies
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Bring in streamlit for UI dev
import streamlit as st
# Bring in watsonx interface
from watsonxlangchain import LangChainInterface

# Setup credentials dictionary
creds = {
    "apikey": "tor1ZjCcjjkupA_XvbLHWvjNuVoy_bDd7iBRogYmtqqI",
    "url": "https://eu-de.ml.cloud.ibm.com"
}
# Create LLM using LangChain
llm = LangChainInterface(
    credentials=creds,
    model="meta-llama/llama-2-70b-chat",
    params={
        "decoding_method": "sample",
        "max_new_tokens": 200,
        "temperature": 0.5,
    },
    project_id="93d14d53-f1e4-4a62-8327-6c2c34f4a413"
)
# This function loads a JSON file of your choosing
@st.cache_resource
def load_json():
    # Update JSON file name here
    json_name = "data.json"
    
    # Load the JSON (update jq_schema depending on JSON structure)
    loader = JSONLoader(
        file_path=json_name,
        jq_schema=".text",   # 👈 change this to match your JSON field
        text_content=False
    )
    
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
# Setup the app title
st.title("Rent")

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
    # Send the prompt to the llm
    response = chain.run(prompt)

    # Extract text safely (if response is a LangChain object)
    reply = response if isinstance(response, str) else response.text

    # Show the LLM response
    st.chat_message("assistant").markdown(reply)

    # Store the LLM response in state
    st.session_state.messages.append({"role": "assistant", "content": reply})
