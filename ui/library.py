import streamlit as st

from utils import get_vector_store


def render_library():
    st.subheader("Indexed Videos")
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
                    c1, c2, c3 = st.columns([1, 3, 1])
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
                    with c3:
                        if st.button("Delete", key=f"del_{record.id}"):
                            with st.spinner("Deleting..."):
                                get_vector_store().delete_document(record.id)
                            st.success(
                                f"Deleted '{metadata.get('filename', 'Unknown Video')}'")
                            st.rerun()  # Refresh the UI instantly
    except Exception as e:
        st.error(
            f"Could not load library. Ensure Qdrant is running. ({e})")
