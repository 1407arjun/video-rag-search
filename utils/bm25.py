from rank_bm25 import BM25Okapi
import re

from utils import Metadata


def combine_text(metadata: Metadata) -> str:
    return f"""
    Visual Description: {metadata.get('visual_description')}
    Audio Transcript: {metadata.get('transcript')}
    Title: {metadata.get('title')}
    Description: {metadata.get('description')}
    """


def rerank_documents(query, docs, top_k=5) -> list[tuple[Metadata, float]]:
    corpus = []
    for doc in docs:
        metadata: Metadata = doc.metadata
        combined_text = combine_text(metadata)
        tokens = re.sub(r'[^\w\s]', '', combined_text.lower()).split()
        corpus.append(tokens)

    bm25 = BM25Okapi(corpus)
    query_tokens = re.sub(r'[^\w\s]', '', query.lower()).split()
    bm25_scores = bm25.get_scores(query_tokens)

    scored_docs = list(zip(docs, bm25_scores))

    # Sort descending by BM25 score
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Take the top k after reranking
    return scored_docs[:top_k]
