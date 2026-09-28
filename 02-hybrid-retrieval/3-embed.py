import os
from dotenv import load_dotenv
from google import genai
load_dotenv()
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from openai import OpenAI

ROOT_DIR=Path.cwd()
DATA_DIR=ROOT_DIR/'data'/'fiqa'
INDEX_DIR=ROOT_DIR/'indexes'/'dense'
client=genai.Client()


def embed_batch(texts:list[str])->np.ndarray:
    """ Embed a batch of texts and return a len(texts),1536 array"""
    response=client.models.embed_content(model="gemini-embeddin-2",contents=texts)
    return np.array([d.embedding for d in response.embeddings[0].values],dtype=np.float32)


def build_index(doc_texts:list[str],batch:int=256)->np.ndarray:
    """ Embed the full corpus in batches with progress bar"""
    chunks=[]
    for i in tqdm(range(0,len(doc_texts),batch),desc="Embedding"):
        chunks.append(embed_batch[doc_texts[i:i+batch]])
    return np.vstack(chunks)

corpus=pd.read_parquet(DATA_DIR/"corpus.parquet")
doc_ids=corpus['_id'].tolist()
doc_texts=[t.strip() or "[empty document]" for t in corpus['text'].tolist()]

embedding_path=INDEX_DIR/"embeddings.npy"
if embedding_path.exists():
    print("Loading embedding from ",embedding_path)

else:
    print(f"Embedding  {len(doc_texts)}")
    doc_embeddings=build_index(doc_texts)
    np.save(embedding_path,doc_embeddings)