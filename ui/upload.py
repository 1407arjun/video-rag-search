import streamlit as st

from utils import run_extraction_pipeline, get_vector_store, combine_text, Metadata


def render_upload():
    st.subheader("1. Upload Video")
    up_file = st.file_uploader("Upload MP4", type=[
                               "mp4"], key=f"video_uploader_{st.session_state.uploader_key}")

    if up_file and st.button("Process Video"):
        run_extraction_pipeline(up_file)


def render_review():
    if st.session_state.processed_video_data:
        st.divider()
        st.subheader("2. Review Extracted Data")
        st.info(
            "Review and edit the AI-generated descriptions below before saving them to the database.")

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
                metadata: Metadata = {
                    **data,
                    "transcript": edited_transcript,
                    "visual_description": edited_visual_description,
                    "title": edited_title,
                    "description": edited_description,
                    "url": edited_url
                }

                combined_content = combine_text(metadata)

                # Save using LangChain Qdrant integration
                get_vector_store().add_document(content=combined_content, metadata=metadata)

                # Clear the session state to reset the UI
                st.session_state.processed_video_data = None
                st.session_state.uploader_key += 1
                st.success(
                    f"Successfully saved '{data['filename']}' to Qdrant!")
                st.rerun()  # Refresh the UI
