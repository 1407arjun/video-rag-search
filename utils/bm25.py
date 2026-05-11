from rank_bm25 import BM25Okapi
import re
import numpy as np

from utils import Metadata


def combine_text(metadata: Metadata) -> str:
    return f"""
    Visual Description: {metadata.get('visual_description')}
    Audio Transcript: {metadata.get('transcript')}
    Title: {metadata.get('title')}
    Description: {metadata.get('description')}
    URL: {metadata.get('url')}
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

    scored_docs = list(zip([doc.metadata for doc in docs], bm25_scores))

    # Sort descending by BM25 score
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    # Take the top k after reranking
    scored_docs = scored_docs[:top_k]

    # Check for distinct gap in scores to determine if we should cut off earlier than top_k
    differences = np.abs(
        np.diff(np.array([score for _, score in scored_docs])))
    max_gap = np.max(differences)
    avg_gap = np.mean(differences)

    # If the largest gap is 2 times larger than the average gap
    if max_gap >= (avg_gap * 2):
        largest_gap_index = np.argmax(differences)
        return scored_docs[:largest_gap_index + 1]
    else:
        return scored_docs
