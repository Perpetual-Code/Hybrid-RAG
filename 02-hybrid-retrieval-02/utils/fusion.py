from collections import defaultdict

from utils.retrievers import BM25Retriever, DenseRetriever

bm25_obj = BM25Retriever()
dense_obj = DenseRetriever()

SMOOTHING_FACTOR_K = 60


def rrf(rankings: list[list[str]], k: int = SMOOTHING_FACTOR_K):
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid_search(
    query: str, bm25: BM25Retriever, dense: DenseRetriever, k: int = 50
) -> list[tuple[str, float]]:
    bm25_ids = [doc_id for doc_id, _ in bm25.bm25_search(query, k=k)]
    # print(bm25_ids)
    dense_ids = [doc_id for doc_id, _ in dense.dense_search(query, k=k)]
    # print(dense_ids)
    return rrf([bm25_ids, dense_ids])[:k]
