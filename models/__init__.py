import os
import streamlit as st

from .llm import OpenAIModel, OpenAIEmbeddingModel
from .asr import ASRModel

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.environ.get("OPENAI_ENDPOINT")

llm = OpenAIModel(api_key=OPENAI_API_KEY, endpoint=OPENAI_ENDPOINT)
asr = ASRModel()
embedding = OpenAIEmbeddingModel(
    api_key=OPENAI_API_KEY, endpoint=OPENAI_ENDPOINT)


@st.cache_resource
def get_llm():
    return llm


@st.cache_resource
def get_asr_model():
    return asr


@st.cache_resource
def get_embedding():
    return embedding
