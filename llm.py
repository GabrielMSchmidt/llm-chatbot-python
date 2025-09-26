import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Create the LLM

llm = ChatGoogleGenerativeAI(
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    model=st.secrets["GOOGLE_MODEL"],
)

# Create the Embedding model

embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    model="gemini-embedding-001"
)