from collections import defaultdict

from utils.retrievers import BM25Retriever, DenseRetriever, load_corpus

# INFO Smoothing Factor
SMOOTHING_FACTOR_K = 60

# INFO Load corpus
corpus = load_corpus()
# INFO Get Doc IDS
docs_id = [t.strip() for t in corpus["_id"].tolist()]

# INFO Initialize class instance
bm25 = BM25Retriever()
dense = DenseRetriever()


# INFO Define Reciprocal Rank Fusion
def rrf(
    rankings: list[list[str]], k: int = SMOOTHING_FACTOR_K
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# INFO Define HYBRID Search
def hybrid_search(query: str, k: int = 10, top_k: int = 50) -> list[tuple[str, float]]:
    bm25_id = [doc_id for doc_id, _ in bm25.bm25_search(query=query, k=top_k)]
    dense_id = [doc_id for doc_id, _ in dense.dense_search(query=query, k=top_k)]
    return rrf([bm25_id, dense_id])[:k]


# INFO Show
def show(label: str, results: list[tuple[str, float]]) -> None:
    print(f"\n{label}\n")
    for i, (doc_id, score) in enumerate(results[:5], 1):
        text = corpus.loc[corpus["_id"] == doc_id, "text"].iloc[0]
        print(f"{i} [{score:.5f} {doc_id} {text[:100]}]")


if __name__ == "__main__":
    query = "where should I park my rainy-day fund?"

    print("Query:", query)

    show("BM25 only", bm25.bm25_search(query, k=5))
    show("Dense only", dense.dense_search(query, k=5))
    show("RRF Only", hybrid_search(query, k=5))
