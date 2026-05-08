from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore

from .llm import OpenAIModel


class VectorStore:
    def __init__(self, qdrant_url: str, collection_name: str, llm: OpenAIModel):
        q_client = QdrantClient(url=qdrant_url)

        collections = q_client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            q_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=1536, distance=models.Distance.COSINE),
            )

        self.store = QdrantVectorStore(
            client=q_client,
            collection_name=COLLECTION_NAME,
            embedding=llm.get_embedding()
        )

    def get_vector_store(self):
        return self.store
