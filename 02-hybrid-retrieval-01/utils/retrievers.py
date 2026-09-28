"""
Polished versions of the BM25 and dense retrievers build in the bm25.py and
embed.py. The retrievers are imported from here so each one can focus on the new idea it
introduces(fusion,reranking, evaluation) instead of re-loading state
"""

import os
from pathlib import Path
from typing import cast

import bm25s
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# INFO Directories
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "fiqa"
BM25_DIR = ROOT_DIR / "Index" / "bm25"
DENSE_DIR = ROOT_DIR / "Index" / "dense"


# INFO client
def create_client(var: str) -> NVIDIAEmbeddings:
    if var not in os.environ:
        load_dotenv()
        os.environ[var] = cast(str, os.getenv(var))

    client = NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b")
    return client


# INFO Load corpus
def load_corpus() -> pd.DataFrame:
    corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
    return corpus


# INFO BM25
class BM25Retriever:
    def __init__(self) -> None:
        self._retriever = bm25s.BM25.load(str(BM25_DIR))
        self._doc_ids = (BM25_DIR / "docs_id.txt").read_text().splitlines()

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        tokens = bm25s.tokenize(query, stopwords="en")
        indices, scores = self._retriever.retrieve(query_tokens=tokens)

        return [
            (self._doc_ids[i], float(scores[0][j]))
            for j, i in enumerate(indices[0].tolist())
        ]


# INFO DENSE


class DenseRetriever:
    def __init__(self) -> None:
        self._client = create_client("NVIDIA_API_KEY")
        self._corpus = load_corpus()
        self._docs_id = [t.strip() for t in self._corpus["_id"].tolist()]
        raw = np.load(DENSE_DIR / "embeddings.npy")
        self._embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    def embed_query(self, query: list[str]) -> np.ndarray:
        response = self._client.embed_documents(query)
        vector = np.asarray(response[0], dtype=np.float32)
        normalized_vector = vector / np.linalg.norm(vector, keepdims=True)

        return normalized_vector

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:

        scores = self._embeddings @ self.embed_query([query])

        top_k = np.argsort(-scores)[:k]

        return [(self._docs_id[i], float(scores[i])) for i in top_k]
