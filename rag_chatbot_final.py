import os
import streamlit as st
import warnings
import logging

# LangChain + RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ----------------------
# API Key from Secrets
# ----------------------
if "GROQ_API_KEY" not in st.secrets:
    st.error("🚫 GROQ_API_KEY is missing from Streamlit secrets!")
    st.stop()

# Streamlit UI Setup
st.set_page_config(page_title="Ask Chatbot!", layout="centered")
st.title("Ask Chatbot! 🤖")

PDF_PATH = "reflexion.pdf"

# Save chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Cache vectorstore creation
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError("🚫 PDF not found. Please ensure 'reflexion.pdf' is in the same folder.")
    
    loader = PyPDFLoader(PDF_PATH)
    return VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders([loader]).vectorstore

# Chat input
prompt = st.chat_input("Ask anything about the PDF...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        vectorstore = load_vectorstore()

        llm = ChatGroq(
            groq_api_key=st.secrets["GROQ_API_KEY"],
            model_name="llama3-8b-8192"
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        result = chain({"query": prompt})
        response = result["result"]

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"🚫 Error: {str(e)}")
