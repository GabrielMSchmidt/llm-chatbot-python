import streamlit as st
import getpass
import os

# Create the LLM
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

llm = ChatGoogleGenerativeAI(
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    model=st.secrets["GOOGLE_MODEL"],
)

# Create the Embedding model

embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    model="gemini-embedding-001"
)