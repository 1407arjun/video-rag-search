import streamlit as st

from utils import get_vector_store


def render_search():
    query = st.text_input(
        "Search by content (e.g., 'someone making a recipe'):")
    if query:
        docs = get_vector_store().similarity_search(query)
        if docs:
            for i, doc in enumerate(docs):
                metadata = doc.metadata
                with st.container(border=True):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image(metadata.get('thumbnail'))
                    with col2:
                        st.subheader(metadata.get('filename'))
                        st.write(
                            f"**Visual:** {metadata.get('visual_description')}")
                        st.caption(f"**Speech:** {metadata.get('transcript')}")
                        if metadata.get('url') is not None:
                            st.markdown(
                                f"[{metadata.get('url')}]({metadata.get('url')})")
        else:
            st.info("No videos found matching your query.")
