import streamlit as st

from utils import get_vector_store, rerank_documents
from .library import render_card


def render_search():
    query = st.text_input("Search anything:")
    if query and len(query.strip()) >= 3:
        docs = get_vector_store().similarity_search(query, retreival_count=10)
        if docs and len(docs) > 0:
            reranked_docs = rerank_documents(query.strip(), docs, top_k=5)
            for i, (metadata, score) in enumerate(reranked_docs):
                render_card(metadata, rank=i+1, score=score)
        else:
            st.info("No videos found matching your query.")
