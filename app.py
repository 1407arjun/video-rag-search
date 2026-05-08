import streamlit as st

# autopep8: off
from dotenv import load_dotenv
load_dotenv()

from ui import render_search, render_upload, render_review, render_library

st.set_page_config(page_title="Video Semantic Search (Qdrant)", layout="wide")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "processed_video_data" not in st.session_state:
    st.session_state.processed_video_data = None

st.title("🎥 AI Video Search (Qdrant + LangChain)")

tab_search, tab_upload, tab_library = st.tabs(
    ["🔍 Search", "⬆️ Upload & Review", "📚 Video Library"])

with tab_search:
    render_search()

with tab_upload:
    render_upload()
    render_review()

with tab_library:
    render_library()
