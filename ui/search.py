import streamlit as st

from utils import get_vector_store, rerank_documents, Metadata


def render_card(metadata: Metadata, rank: int):
    st.subheader(f"Rank #{rank}")
    st.caption(metadata.get('title', 'Untitled Video'))
    if metadata.get('url') is not None:
        st.markdown(
            f"[{metadata.get('url')}]({metadata.get('url')})")
    st.image(metadata.get('thumbnail'), width=250)


def render_search():
    query = st.text_input("Search anything:")
    if query and len(query.strip()) >= 3:
        docs = get_vector_store().similarity_search(query, retreival_count=10)
        if docs and len(docs) > 0:
            reranked_docs = rerank_documents(query.strip(), docs, top_k=5)
            with st.container(border=True):
                columns = st.columns(len(reranked_docs))
                for i, (metadata, score) in enumerate(reranked_docs):
                    with columns[i]:
                        render_card(metadata, rank=i+1)
        else:
            st.info("No videos found matching your query.")
