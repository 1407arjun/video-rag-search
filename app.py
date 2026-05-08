import os
import streamlit as st
import tempfile

from utils.vector_store import VectorStore
from utils.asr import ASRModel
from utils.llm import OpenAIModel
from utils.vector_store import Metadata
from data_extraction import DataExtractor

from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.environ.get("OPENAI_ENDPOINT")

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "video_contents"

llm = OpenAIModel(api_key=OPENAI_API_KEY, endpoint=OPENAI_ENDPOINT)
asr = ASRModel()
store = VectorStore(url=QDRANT_URL, api_key=QDRANT_API_KEY,
                    collection_name=COLLECTION_NAME, llm=llm)


def run_extraction_pipeline(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    data_extractor = DataExtractor(video_path, llm=llm, asr=asr)

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
