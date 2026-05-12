import os
import streamlit as st

from .data_extraction import DataExtractor
from .vector_store import VectorStore, Metadata
from .bm25 import rerank_documents, combine_text
from .download import download_video, VideoData


class SessionData(Metadata):
    frames: list[str]


def run_extraction_pipeline(video_data: VideoData):
    video_path = video_data["filepath"]
    data_extractor = DataExtractor(video_path)

    try:
        transcript, caption, thumbnails = data_extractor.extract_data()
        # Save to session state for the review UI
        metadata: SessionData = {
            "transcript": transcript,
            "visual_description": caption,
            "title": f"Video by {video_data['uploader'] or 'Unknown'}",
            "description": video_data["caption"],
            "frames": thumbnails,
            "url": video_data["original_url"]
        }
        st.session_state.processed_video_data = metadata
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


@st.cache_resource
def get_vector_store():
    store = VectorStore(url=st.secrets.qdrant.url,
                        api_key=st.secrets.qdrant.api_key, collection_name="video_contents")
    return store
