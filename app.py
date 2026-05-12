import streamlit as st

from ui import render_search, render_upload, render_review, render_library

st.set_page_config(page_title="Semantic Video Search", layout="wide")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "processed_video_data" not in st.session_state:
    st.session_state.processed_video_data = None

st.title("Semantic Video Search")
st.subheader("Making reels searchable with RAG and BM25")

tab_search, tab_upload, tab_library = st.tabs(
    ["Search", "Ingest and Review", "Video Library"])

with tab_search:
    render_search()

with tab_upload:
    render_upload()
    render_review()

with tab_library:
    render_library()
