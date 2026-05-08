import os
import tempfile
import streamlit as st

from .data_extraction import DataExtractor
from .vector_store import VectorStore, Metadata

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "video_contents"

store = VectorStore(url=QDRANT_URL, api_key=QDRANT_API_KEY,
                    collection_name=COLLECTION_NAME)


def run_extraction_pipeline(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    data_extractor = DataExtractor(video_path)

    try:
        transcript, caption, thumbnail = data_extractor.extract_data()
        # Save to session state for the review UI
        metadata: Metadata = {
            "filename": uploaded_file.name,
            "transcript": transcript,
            "visual_description": caption,
            "title": "",
            "description": "",
            "thumbnail": thumbnail,
            "url": ""
        }
        st.session_state.processed_video_data = metadata
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


@st.cache_resource
def get_vector_store():
    return store
