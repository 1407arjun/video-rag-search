import streamlit as st

from utils import get_vector_store, rerank_documents


def render_search():
    query = st.text_input(
        "Search by content (e.g., 'someone making a recipe'):")
    if query:
        docs = get_vector_store().similarity_search(query, retreival_count=10)
        if docs:
            reranked_docs = rerank_documents(query, docs, top_k=5)
            for i, (metadata, score) in enumerate(reranked_docs):
                with st.container(border=True):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image(metadata.get('thumbnail'))
                    with col2:
                        st.subheader(
                            f"{metadata.get('title')} (Rank: #{i+1}, Score: {score:.2f})")
                        st.write(
                            f"**Caption:** {metadata.get('description')}")
                        st.caption(
                            f"**Visual:** {metadata.get('visual_description')}")
                        st.caption(f"**Speech:** {metadata.get('transcript')}")
                        if metadata.get('url') is not None:
                            st.markdown(
                                f"[{metadata.get('url')}]({metadata.get('url')})")
        else:
            st.info("No videos found matching your query.")
