import streamlit as st
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore

from .llm import get_embedding_model

QDRANT_URL = os.environ.get("QDRANT_URL")
COLLECTION_NAME = "video_contents"

@st.cache_resource
def get_vector_store():
    # Initialize Qdrant Client
    q_client = QdrantClient(url=QDRANT_URL)
    
    # Create collection if it doesn't exist
    collections = q_client.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        q_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        )
    
    return QdrantVectorStore(
        client=q_client,
        collection_name=COLLECTION_NAME,
        embedding=get_embedding_model()
    )