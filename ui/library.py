import streamlit as st

from utils import get_vector_store, Metadata


def render_card(metadata: Metadata, id: str | None = None, rank: int | None = None, score: float | None = None):
    with st.container(border=True):
        c1, c2 = st.columns([1, 4])
        with c1:
            st.image(metadata.get('thumbnail'), width=250)
        with c2:
            title = metadata.get('title', 'Untitled Video') + (
                f" (Rank: #{rank}, Score: {score:.2f})" if rank is not None and score is not None else "")
            st.subheader(title)
            if metadata.get('url') is not None:
                st.markdown(
                    f"[{metadata.get('url')}]({metadata.get('url')})")
            st.caption(
                f"**Caption:** {metadata.get('description', '')}")
            st.caption(
                f"**Visuals:** {metadata.get('visual_description', '')}")
            st.caption(
                f"**Transcript:** {metadata.get('transcript', '')}")
            if id is not None:
                if st.button("Delete", key=f"del_{id}"):
                    with st.spinner("Deleting..."):
                        get_vector_store().delete_document(id)
                    st.success(
                        f"Deleted '{metadata.get('title')}'")
                    st.rerun()  # Refresh the UI instantly


def render_library():
    st.subheader("Ingested Videos")
    try:
        records = get_vector_store().list_documents()

        if not records:
            st.info("No videos in the database yet.")
        else:
            for record in records:
                metadata = record.payload.get("metadata", {})
                render_card(metadata, id=record.id)
    except Exception as e:
        st.error(
            f"Could not load library. Ensure Qdrant is running. ({e})")
