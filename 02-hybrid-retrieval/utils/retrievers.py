import os
import numpy as np
import pandas as pd
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from dotenv import load_dotenv
from typing import cast
from pathlib import Path
import bm25s

load_dotenv()
os.environ['NVIDIA_API_KEY']=cast(str,os.getenv('NVIDIA_API_KEY'))
ROOT_DIR=Path(__file__).resolve().parent
DATA_DIR=ROOT_DIR/"data"/"fiqa"
BM25_DIR=ROOT_DIR/"indexes"/"bm25"
DENSE_DIR=ROOT_DIR/"indexes"/"dense"
EMBEDDING_MODEL="nvidia/nemotron-3-embed-1b"

client=NVIDIAEmbeddings(model=EMBEDDING_MODEL)

def load_corpus():
    return pd.read_parquet(DATA_DIR/"corpus.parquet")

class DenseRetriever:
    def __init__(self) -> None:
        corpus=load_corpus()
        self._doc_ids=corpus["_id"].tolist()
        raw=np.load(DENSE_DIR/"embeddings.npy")
        self._embeddings=raw/np.linalg.norm(raw,axis=1,keepdims=True)

    def _embed_query(self,query:str)->np.ndarray:
        response=client.embed_documents(texts=[query])
        vector=np.asarray(response)
        return vector/np.linalg.norm(vector,axis=1,keepdims=True)


    def search(self,query:str,k:int=10)->list[tuple[str,float]]:
        scores=self._embeddings@self._embed_query(query)
        top_k=np.argsort(-scores)[:k]
        return[(self._doc_ids[i],float(scores[i])) for i in top_k]


class BM25Retriever:
    def __init__(self) -> None:
        self._retriever=bm25s.BM25.load(BM25_DIR)
        self._doc_ids=(BM25_DIR/"doc_ids.txt").read_text().splitlines()

    def search(self,query:str,k:int=10)->list[tuple[str,float]]:
        tokens=bm25s.tokenize([query],stopwords="en")
        indices,scores=self._retriever.retrieve(tokens,k=k)
        return [(self._doc_ids[i],float(scores[0][j])) for j,i in enumerate(indices[0].tolist())]