import os
from collections import defaultdict

import cohere
from dotenv import load_dotenv
from utils.retrievers import BM25Retriever, DenseRetriever, load_corpus

load_dotenv()

SMOOTHING_FACTOR_K = 60
corpus = load_corpus()
bm25 = BM25Retriever()
dense = DenseRetriever()

print(len(corpus))
corpus_by_id = corpus.set_index("_id")
# print(corpus_by_id.head())


co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
RERANK_MODEL = "rerank-v4.0-fast"


def rrf(rankings: list[list[str]], k: int = SMOOTHING_FACTOR_K):
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid_search(Query: str, k: int = 50):
    bm25_ids = [doc_id for doc_id, _ in bm25.bm25_search(query=Query, k=k)]
    # print(bm25_ids)
    dense_ids = [doc_id for doc_id, _ in dense.dense_search(query=Query, k=k)]
    # print(dense_ids)
    return rrf([bm25_ids, dense_ids])


# print(candidate_texts[:10])


def search_reranked(query: str, k: int = 10):
    candidate_ids = hybrid_search(query)
    true_ids = [doc_id for doc_id, _ in candidate_ids]
    candidate_texts = [corpus_by_id.loc[doc_id, "text"] for doc_id in true_ids]

    response = co.rerank(
        model=RERANK_MODEL, query=query, documents=candidate_texts, top_n=k
    )

    return [(true_ids[r.index], r.relevance_score) for r in response.results]


def show(label: str, results: list[tuple[str, float]]):
    print(label)
    for i, (doc_id, score) in enumerate(results[:5], 1):
        text = corpus_by_id.loc[doc_id, "text"]
        print(f"{i} [{score:.3f} {text[:70]}]")


if __name__ == "__main__":
    Query = "Where should I park my rainy-day funds?"
    print(Query)

    show("Hybrid", search_reranked(Query))
