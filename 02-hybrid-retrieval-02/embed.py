import os
import random
import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from tqdm import tqdm

# NOTE DIRECTORY PATH
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "fiqa"
INDEX_DIR = ROOT_DIR / "indexes"
CHECKPOINT_DIR = INDEX_DIR

# print(ROOT_DIR)

# NOTE LOAD corpus data
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
docs_text = [t.strip() or "[EMPTY DOCUMENT]" for t in corpus["text"].tolist()]
docs_id = [i.strip() for i in corpus["_id"].tolist()]


# NOTE Create NVIDIA Embedding client
def create_client():
    load_dotenv()
    os.environ["NVIDIA_API_KEY"] = cast(str, (os.getenv("NVIDIA_API_KEY")))
    client = NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b", trucate="NONE")
    return client


# NOTE Create Embeddings
def create_embed(text_to_embed: list[str], max_retries: int = 5) -> np.ndarray:
    client = create_client()
    for attempt in range(max_retries):
        try:
            response = client.embed_documents(texts=text_to_embed)
            vector = np.asarray(response, dtype=np.float32)
            if len(response[0]) != vector.shape[1]:
                raise ValueError(
                    f"Expected {len(response[0])} but recieved {vector.shape[1]}"
                )
            return vector
        except ReferenceError as error:
            if attempt == max_retries - 1:
                delay = (2 * attempt) + random.random()
                print(f"{error} occured.. retrying in {delay}")
                time.sleep(delay)
    raise ValueError("Something went wrong")


# NOTE Build Index
def build_index(texts: list[str], batch_size: int = 256) -> np.ndarray:
    chunks: list[np.ndarray] = []
    start_index: int = 0
    checkpoint_path = CHECKPOINT_DIR / "embeddings.npz"
    if checkpoint_path.exists():
        checkpoint = np.load(checkpoint_path)
        chunks = [checkpoint["embeddings"]]
        start_index = int(checkpoint["next_index"])
    for i in tqdm(range(start_index, len(texts), batch_size), desc="Embedding"):
        batch_chunk = texts[i : i + batch_size]
        vector = create_embed(batch_chunk)
        chunks.append(vector)
        np.savez_compressed(
            checkpoint_path,
            embeddings=np.vstack(chunks),
            next_index=int(i + len(batch_chunk)),
        )

    final_embedding = np.vstack(chunks)
    return final_embedding


# NOTE Dense Search
def dense_search(query_text: str, embed_doc: np.ndarray, k: int = 10):
    client = create_client()
    embed_query = client.embed_documents([query_text])
    query_vector = np.asarray(embed_query, dtype=np.float32)
    normalized_embed_doc = query_vector / np.linalg.norm(query_vector, keepdims=True)
    scores = embed_doc @ normalized_embed_doc
    top_k = np.argsort(-scores)[:k]
    return [(docs_id[i], float(scores[i])) for i in top_k]


# NOTE Main
def main():
    Query = "Where should I park my rainy-day funds?"
    if (CHECKPOINT_DIR / "embeddings.npy").exists():
        final_embedding = np.load(CHECKPOINT_DIR / "embeddings.npy")
        print("Embedding loaded")
    else:
        CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        print("Embedding documents")
        final_embedding = build_index(docs_text)
        np.save(CHECKPOINT_DIR / "embedding.npy", final_embedding)
        final_embedding_normalized = final_embedding / np.linalg.norm(
            final_embedding, keepdims=True
        )

    for i, (doc_id, score) in enumerate(
        dense_search(embed_doc=final_embedding_normalized, query_text=Query, k=5), 1
    ):
        text = corpus.loc[corpus["_id"] == doc_id, "text"].iloc[0]
        print(f"{i} {doc_id}:[{score:3f}:{text[:100]}]")


if __name__ == "__main__":
    main()
