import os
from pathlib import Path
from typing import cast

import bm25s
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_nvidia import NVIDIAEmbeddings

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "fiqa"
BM25_DIR = ROOT_DIR / "indexes" / "bm25"
DENSE_DIR = ROOT_DIR / "indexes" / "dense"
EMBEDDING_PATH = DENSE_DIR / "embeddings.npy"
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")


# INFO Create Client
def create_client():
    load_dotenv()
    os.environ["NVIDIA_API_KEY"] = cast(str, os.getenv("NVIDIA_API_KEY"))
    client = NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b", trucate="NONE")
    return client


# INFO Load Coprus
def load_corpus():
    return pd.read_parquet(DATA_DIR / "corpus.parquet")


# INFO BM25Retriever Class
class BM25Retriever:
    def __init__(self) -> None:
        self.retriever = bm25s.BM25().load(BM25_DIR)
        self.docs_id = (BM25_DIR / "docs.txt").read_text().splitlines()

    def bm25_search(self, query: str, k: int = 10):
        query_token = bm25s.tokenize(query, stopwords="en")
        indices, scores = self.retriever.retrieve(query_tokens=query_token, k=k)
        return [
            (self.docs_id[i], scores[0][j]) for j, i in enumerate(indices[0].tolist())
        ]


# INFO Dense Retriever Class
class DenseRetriever:
    def __init__(self) -> None:
        self.client = create_client()
        corpus = load_corpus()
        raw = np.load(EMBEDDING_PATH)
        self._docs_id = corpus["_id"].tolist()
        self.normalized_embedding = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    # INFO Embed Query
    def embed_query(self, query: str) -> np.ndarray:
        response = self.client.embed_documents([query])[0]
        query_vector = np.asarray(response, dtype=np.float32)
        normalized_query = query_vector / np.linalg.norm(query_vector, keepdims=True)

        return normalized_query

    # INFO DENSE Search
    def dense_search(self, query: str, k: int = 10):
        normalized_query = self.embed_query(query)
        scores = self.normalized_embedding @ normalized_query
        top_k = np.argsort(-scores)[:k]
        return [(self._docs_id[i], float(scores[i])) for i in top_k]
