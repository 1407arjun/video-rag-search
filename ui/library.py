import streamlit as st

from utils import get_vector_store


def render_library():
    st.subheader("Indexed Videos")
    if st.button("Refresh Library"):
        try:
            records = get_vector_store().list_documents()

            if not records:
                st.info("No videos in the database yet.")
            else:
                for record in records:
                    payload = record.payload
                    # LangChain Qdrant integration nests your metadata inside a "metadata" key in the payload
                    metadata = payload.get("metadata", {})
                    with st.container(border=True):
                        c1, c2 = st.columns([1, 4])
                        with c1:
                            st.image(metadata.get('thumbnail'))
                        with c2:
                            st.markdown(
                                f"**🎬 {metadata.get('filename', 'Unknown Video')}**")
                            st.caption(
                                f"**Visuals:** {metadata.get('visual_description', '')[:150]}...")
                            st.caption(
                                f"**Transcript:** {metadata.get('transcript', '')[:150]}...")
                            if metadata.get('url') is not None:
                                st.markdown(
                                    f"[{metadata.get('url')}]({metadata.get('url')})")
        except Exception as e:
            st.error(
                f"Could not load library. Ensure Qdrant is running. ({e})")
