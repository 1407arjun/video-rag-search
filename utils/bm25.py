from rank_bm25 import BM25Okapi
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

from utils import Metadata

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


def combine_text(metadata: Metadata) -> str:
    return f"""
    Visual Description: {metadata.get('visual_description')}
    Audio Transcript: {metadata.get('transcript')}
    Title: {metadata.get('title')}
    Description: {metadata.get('description')}
    URL: {metadata.get('url')}
    """


def preprocess_text(text: str) -> list[str]:
    # Lowercase and remove non-alphanumeric characters except whitespace, underscores, and hyphens
    text = re.sub(r'[^\w\s\-]', '', text.lower())

    # Tokenization
    tokens = word_tokenize(text)

    # Stop words and punctuation removal
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    tokens = [t for t in tokens if t not in stop_words and t not in punctuation]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def rerank_documents(query, docs, top_k=5) -> list[tuple[Metadata, float]]:
    corpus = []
    for doc in docs:
        metadata: Metadata = doc.metadata
        combined_text = combine_text(metadata)
        tokens = preprocess_text(combined_text)
        corpus.append(tokens)

    bm25 = BM25Okapi(corpus)
    query_tokens = preprocess_text(query)
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
