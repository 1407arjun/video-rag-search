import streamlit as st
import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


class OpenAIModel:
    def __init__(self):
        self._client = OpenAI(api_key=OPENAI_API_KEY)
        self._embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small")

    def call_llm(self, messages: list) -> str:
        response = self._client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def get_embedding(self):
        return self._embedding_model
