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


def _set_env(var: str):
    if var not in os.environ:
        load_dotenv()
        os.environ[var] = cast(str, os.getenv(var))
    else:
        print("Already set")


def create_client():
    _set_env("NVIDIA_API_KEY")
    client = NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b", trucate="NONE")
    return client


# INFO : Set the all directories
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "fiqa"
INDEX_DIR = ROOT_DIR / "indexes" / "dense"
CHECKPOINT = INDEX_DIR

# INFO : Load the data and split the id and text into a flat list  and verify the length of id and texts
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
docs_id = [doc_id.strip() for doc_id in corpus["_id"].tolist()]
docs_text = [t.strip() or "[Empty document]" for t in corpus["text"].tolist()]
CHECKPOINT.mkdir(exist_ok=True, parents=True)


# INFO : Create the function the splits the complete doc text in chunk that can be passed to the embed function
def build_index(client: NVIDIAEmbeddings, texts: list[str], batch_size=256):
    chunks: list[np.ndarray] = []
    start_index: int = 0
    checkpoint_path = CHECKPOINT / "embeddings.npz"
    if checkpoint_path.exists():
        checkpoint = np.load(checkpoint_path)
        chunks = [checkpoint["embedding"]]
        start_index = int(checkpoint["next_index"])

    # NOTE if error occurs change i to start_index
    for i in tqdm(range(start_index, len(texts), batch_size), desc="Embedding"):
        chunk = texts[i : i + batch_size]
        vector = embed_with_retry(client=client, text=chunk)
        chunks.append(vector)

        np.savez_compressed(
            checkpoint_path,
            embedding=np.vstack(chunks),
            next_index=start_index + len(chunk),
        )

    final_embedding = np.vstack(chunks)

    if final_embedding.shape[0] != len(texts):
        raise ValueError(f"Expected {len(texts)}\n recieved {final_embedding.shape[0]}")

    return final_embedding


# INFO Create the embed function with retry logic
def embed_with_retry(
    text: list[str], client: NVIDIAEmbeddings, max_retries: int = 5
) -> np.ndarray:

    for attempt in range(max_retries):
        try:
            vector = client.embed_documents(text)
            if np.asarray(vector).shape[0] != len(text):
                raise ValueError(
                    f"Expected {len(text)}\n recived {np.asarray(vector).shape[0]}"
                )
            return np.asarray(vector)
        except ReferenceError as e:
            if attempt == max_retries - 1:
                raise ValueError(f"Encountered error {e}")
            delay = (2**attempt) + random.random()
            print(f"Encountered error {e}\n retrying  again in {delay}")
            time.sleep(delay)
    raise ValueError("Error occured")


# INFO Dense search Function
def dense_search(
    doc_embed_norm: np.ndarray, query: str, client: NVIDIAEmbeddings, k: int = 5
) -> list[tuple[str, float]]:
    embed_query = embed_with_retry([query], client=client)[0]
    embed_query_normalized = embed_query / np.linalg.norm(embed_query, keepdims=True)
    scores = doc_embed_norm @ embed_query_normalized
    top_k = np.argsort(-scores)[:k]

    return [(docs_id[i], (float(scores[i]))) for i in top_k]


def main():
    client = create_client()
    Final_Embeddings_path = INDEX_DIR / "embeddings.npy"
    query = "where should I park my rainy funds"
    if Final_Embeddings_path.exists():
        doc_embedding = np.load(Final_Embeddings_path)
        print("Embeddings loaded")
    else:
        doc_embedding = build_index(client=client, texts=docs_text)
        np.save(Final_Embeddings_path, doc_embedding)
        print("Embeddings saved")

    # INFO Creating the normalized embedding
    doc_embedding_normalized = doc_embedding / np.linalg.norm(
        doc_embedding, axis=1, keepdims=True
    )

    # NOTE For debugging
    # response = embed_with_retry(text=[query],client=client)[0]
    # query_normalized=response/np.linalg.norm(response,keepdims=True)

    # print(f"Final Embedding Shape:{doc_embedding_normalized.shape}\n Query Shape: {query_normalized.shape}")

    # # result=doc_embedding_normalized @ query_normalized
    # result=doc_embedding_normalized @ query_normalized

    for i, (doc_id, score) in enumerate(
        dense_search(
            client=client, query=query, doc_embed_norm=doc_embedding_normalized
        ),
        1,
    ):
        text = corpus.loc[corpus["_id"] == doc_id, "text"].iloc[0]
        print(f"{i} [{score:.3f}] {doc_id} {text[:100]}")


if __name__ == "__main__":
    main()
