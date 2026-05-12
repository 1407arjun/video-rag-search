import streamlit as st
import re

from utils import run_extraction_pipeline, get_vector_store, combine_text, download_video, Metadata, SessionData

pattern = r"^(https?://)?(www\.)?instagram\.com/reels?/[\w-]+/?"


def render_upload():
    st.subheader("1. Download Instagram Reel")
    video_url = st.text_input(
        "Enter Instagram Reel URL",
        placeholder="https://www.instagram.com/reels/...",
        key=f"url_input_{st.session_state.uploader_key}"
    )
    if video_url:
        if re.match(pattern, video_url) is None:
            st.error(
                "This does not look like a valid Instagram Reel URL. Please check the link and try again.")
        else:
            if st.button("Download & Process"):
                try:
                    with st.spinner("Downloading from Instagram..."):
                        video_data = download_video(video_url)
                        st.success(
                            "Downloaded successfully! Processing extraction pipeline...")

                    with st.spinner("Running visual and audio analysis..."):
                        run_extraction_pipeline(video_data)

                except Exception as e:
                    st.error(
                        f"Failed to download or process reel. Instagram might be rate-limiting. Details: {e}")


def render_review():
    if st.session_state.processed_video_data:
        st.divider()
        st.subheader("2. Review Extracted Data")
        st.info(
            "Review and edit the AI-generated descriptions below before saving them to the vector database.")

        data: SessionData = st.session_state.processed_video_data

        col_thumb, col_edit = st.columns([1, 2])
        with col_thumb:
            frames = data["frames"]
            COLS = 3
            ROWS = len(frames) // COLS + 1

            for i in range(ROWS):
                cols = st.columns(COLS)
                for j in range(COLS):
                    idx = i * COLS + j
                    if idx < len(frames):
                        cols[j].image(frames[idx], caption=f"Frame {idx+1}")
        with col_edit:
            edited_visual_description = st.text_area(
                "Visual Description (VLM)", value=data["visual_description"], height=200)
            edited_transcript = st.text_area(
                "Audio Transcript (Whisper)", value=data["transcript"], height=100)
            edited_url = st.text_input("URL", value=data["url"])
            edited_title = st.text_input("Title", value=data["title"])
            edited_description = st.text_area(
                "Description", value=data["description"], height=200)

        if st.button("Confirm & Save to Vector Store", type="primary"):
            with st.spinner("Saving to Qdrant..."):
                metadata: Metadata = {
                    "transcript": edited_transcript,
                    "visual_description": edited_visual_description,
                    "title": edited_title,
                    "description": edited_description,
                    "url": edited_url,
                    # Use the first frame as thumbnail
                    "thumbnail": data["frames"][0],
                }

                combined_content = combine_text(metadata)

                # Save using LangChain Qdrant integration
                get_vector_store().add_document(content=combined_content, metadata=metadata)

                # Clear the session state to reset the UI
                st.session_state.processed_video_data = None
                st.session_state.uploader_key += 1
                st.success(
                    f"Successfully saved '{data['title']}' to Qdrant!")
                st.rerun()  # Refresh the UI
