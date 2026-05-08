import os
import streamlit as st

from .llm import OpenAIModel, OpenAIEmbeddingModel
from .asr import ASRModel

asr = ASRModel()


@st.cache_resource
def get_llm():
    llm = OpenAIModel(api_key=st.secrets.llm.api_key,
                      endpoint=st.secrets.llm.endpoint)
    return llm


@st.cache_resource
def get_asr_model():
    return asr


@st.cache_resource
def get_embedding():
    embedding = OpenAIEmbeddingModel(
        api_key=st.secrets.llm.api_key, endpoint=st.secrets.llm.endpoint)
    return embedding
