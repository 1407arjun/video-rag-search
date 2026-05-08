from typing import TypedDict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from models import get_embedding


class Metadata(TypedDict):
    filename: str
    transcript: str
    visual_description: str
    title: str
    description: str
    thumbnail: str


class VectorStore:
    def __init__(self, url: str, api_key: str, collection_name: str):
        q_client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name

        collections = q_client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            q_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=1536, distance=models.Distance.COSINE),
            )

        self.client = q_client
        self.store = QdrantVectorStore(
            client=q_client,
            collection_name=self.collection_name,
            embedding=get_embedding().get_embedding_model()
        )

    def get_vector_store(self):
        return self.store

    def similarity_search(self, query, retreival_count=5):
        return self.store.similarity_search(query, k=retreival_count)

    def add_document(self, content: str, metadata: Metadata):
        self.store.add_documents(
            [Document(page_content=content, metadata=metadata)])
        
    def delete_document(self, document_id: str):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[document_id]),
        )

    def list_documents(self) -> list[Metadata]:
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=100,  # Adjust limit or implement pagination if library grows large
            with_payload=True,
            with_vectors=False  # We don't need to download the giant arrays, just the metadata
        )
        return records
