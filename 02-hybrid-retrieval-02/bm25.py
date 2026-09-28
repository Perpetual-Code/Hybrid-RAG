from pathlib import Path

import bm25s
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "fiqa"
INDEX_DIR = ROOT_DIR / "indexes" / "bm25"

# NOTE Load corpus
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
# NOTE Docs text and Docs Id
docs_text = [t.strip() for t in corpus["text"].tolist()]
docs_id = [i.strip() for i in corpus["_id"].tolist()]

# NOTE Check if index is saved
if (INDEX_DIR / "params.index.json").exists():
    # NOTE Load saved retriever
    retriever = bm25s.BM25().load(INDEX_DIR, load_corpus=False)
else:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # NOTE Create Tokens
    tokens = bm25s.tokenize(docs_text, stopwords="en")
    # NOTE Create Retriever
    retriever = bm25s.BM25()
    retriever.index(tokens)
    # NOTE Save Retriever
    retriever.save(str(INDEX_DIR), corpus=docs_id)

file = INDEX_DIR / "docs.txt"
if not file.exists():
    file.write_text("\n".join(docs_id))


# NOTE Create Bm25Search
def bm25search(query: str, k: int = 10) -> list[tuple[str, float]]:
    query_token = bm25s.tokenize(query, stopwords="en")
    # INFO To get docs instead of doc ID, set corpus=corpus parameter,refer to the documentation
    indices, scores = retriever.retrieve(query_tokens=query_token, k=k)
    return [
        (docs_id[i], float(scores[0][j])) for j, i in enumerate(indices[0].tolist())
    ]


# NOTE Main Function
def main():
    query = "where should I park my rainy-day funds?"
    print(query)
    for i, (indices, score) in enumerate(bm25search(query=query, k=5), 1):
        text = corpus.loc[corpus["_id"] == indices, "text"].iloc[0]
        print(f"{i} [{score:3f} {text[:100]}]")


# NOTE Call Main
if __name__ == "__main__":
    main()
