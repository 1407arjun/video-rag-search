import streamlit as st
import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

@st.cache_resource
def get_llm_client():
    # Initialize LLM client
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client

@st.cache_resource
def get_embedding_model():
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    return embeddings_model
