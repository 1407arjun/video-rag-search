import streamlit as st
import whisper


@st.cache_resource
def get_asr_model():
    # Initialize ASR client
    model = whisper.load_model("base")
    return model
