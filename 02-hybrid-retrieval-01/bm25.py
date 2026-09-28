from pathlib import Path

import bm25s
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "fiqa"
INDEX_DIR = ROOT_DIR / "Index" / "bm25"

corpus_data = pd.read_parquet(DATA_DIR / "corpus.parquet")
docs_id = corpus_data["_id"].tolist()
docs_text = corpus_data["text"].tolist()

INDEX_DIR.mkdir(parents=True, exist_ok=True)


if (INDEX_DIR / "params.index.json").exists():
    print("Index already saved.....\n Loading retriever...")
    retriever = bm25s.BM25.load(str(INDEX_DIR), load_corpus=False)
else:
    tokens = bm25s.tokenize(docs_text, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(tokens)
    retriever.save(INDEX_DIR, corpus=docs_id)

file = INDEX_DIR / "docs_id.txt"
if not file.exists():
    file.write_text("\n".join(docs_id))


def bm25_search(
    query: str,
    docs_id: list[str],
    retriever: bm25s.BM25,
    k: int = 10,
) -> list[tuple[int, float]]:
    query_token = bm25s.tokenize([query], stopwords="en")

    indices, scores = retriever.retrieve(query_token, k=k)

    return [
        (docs_id[i], float(scores[0][j])) for j, i in enumerate(indices[0].tolist())
    ]


def main():
    query = "where should I park my rainy-day fund?"
    print(f"Query:\n{query}")
    for i, (doc_id, score) in enumerate(
        bm25_search(docs_id=docs_id, query=query, retriever=retriever, k=5), 1
    ):
        text = corpus_data.loc[corpus_data["_id"] == doc_id, "text"].iloc[0]
        print(f"{i}  [{score:.2f}] {doc_id} {text[:80]}")


if __name__ == "__main__":
    main()
