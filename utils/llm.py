from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings


class OpenAIModel:
    def __init__(self, api_key: str, endpoint: str):
        self._client = AzureOpenAI(
            api_version="2025-03-01-preview",
            azure_endpoint=endpoint,
            api_key=api_key
        )
        self._embedding_model = AzureOpenAIEmbeddings(
            api_version="2025-03-01-preview",
            model="text-embedding-3-small",
            azure_endpoint=endpoint,
            api_key=api_key
        )

    def call_llm(self, messages: list) -> str:
        response = self._client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def get_embedding(self):
        return self._embedding_model
