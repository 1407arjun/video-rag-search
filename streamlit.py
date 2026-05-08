import streamlit as st
from app import run_extraction_pipeline, get_vector_store
from utils.vector_store import Metadata

st.set_page_config(page_title="Video Semantic Search (Qdrant)", layout="wide")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "processed_video_data" not in st.session_state:
    st.session_state.processed_video_data = None

st.title("🎥 AI Video Search (Qdrant + LangChain)")

tab_search, tab_upload, tab_library = st.tabs(
    ["🔍 Search", "⬆️ Upload & Review", "📚 Video Library"])

with tab_search:
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

# --- TAB 2: UPLOAD & REVIEW ---
with tab_upload:
    st.subheader("1. Upload Video")
    up_file = st.file_uploader("Upload MP4", type=[
                               "mp4"], key=f"video_uploader_{st.session_state.uploader_key}")

    if up_file and st.button("Process Video"):
        run_extraction_pipeline(up_file)

    if st.session_state.processed_video_data:
        st.divider()
        st.subheader("2. Review Extracted Data")
        st.info(
            "Review and edit the AI-generated descriptions below before saving them to the database.")

        # Use text_area so you can edit the AI's output before saving!
        data: Metadata = st.session_state.processed_video_data

        col_thumb, col_edit = st.columns([1, 2])
        with col_thumb:
            st.image(data["thumbnail"], caption="Captured Thumbnail")
        with col_edit:
            edited_visual_description = st.text_area(
                "Visual Description (VLM)", value=data["visual_description"], height=200)
            edited_transcript = st.text_area(
                "Audio Transcript (Whisper)", value=data["transcript"], height=100)
            edited_url = st.text_input("URL (if any)", value=data["url"])
            edited_title = st.text_input("Title (if any)", value=data["title"])
            edited_description = st.text_area(
                "Description (if any)", value=data["description"], height=100)

        if st.button("Confirm & Save to Vector Store", type="primary"):
            with st.spinner("Saving to Qdrant..."):
                combined_content = f"""
                Visual Description: {edited_visual_description}
                Audio Transcript: {edited_transcript}
                Title: {edited_title}
                Description: {edited_description}
                """

                metadata: Metadata = {
                    **data,
                    "transcript": edited_transcript,
                    "visual_description": edited_visual_description,
                    "title": edited_title,
                    "description": edited_description,
                    "url": edited_url
                }

                # Save using LangChain Qdrant integration
                get_vector_store().add_document(content=combined_content, metadata=metadata)

                # Clear the session state to reset the UI
                st.session_state.processed_video_data = None
                st.session_state.uploader_key += 1
                st.success(
                    f"Successfully saved '{data['filename']}' to Qdrant!")
                st.rerun()  # Refresh the UI

# --- TAB 3: LIBRARY ---
with tab_library:
    st.subheader("Indexed Videos")
    if st.button("Refresh Library"):
        # Use Qdrant's scroll API to get all points in the collection
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
