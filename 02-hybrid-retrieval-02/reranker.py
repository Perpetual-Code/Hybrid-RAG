from dotenv import load_dotenv
from utils.fusion import hybrid_search
from utils.reranker import ReRanker
from utils.retrievers import BM25Retriever, DenseRetriever, load_corpus

load_dotenv()


corpus = load_corpus()
bm25_obj = BM25Retriever()
dense_obj = DenseRetriever()

RERANK_MODEL = "rerank-v4.0-fast"
# print(len(corpus))
corpus_by_id = corpus.set_index("_id")
# print(corpus_by_id.head())


# print(candidate_texts[:10])


reranker = ReRanker(ReRankModel=RERANK_MODEL)


def search_reranked(query: str, k: int = 10):
    candidate_ids = hybrid_search(query, bm25=bm25_obj, dense=dense_obj)
    true_ids = [doc_id for doc_id, _ in candidate_ids]
    candidate_texts = [str(corpus_by_id.loc[doc_id, "text"]) for doc_id in true_ids]
    candidates = [(doc_id, text) for doc_id, text in zip(true_ids, candidate_texts)]
    return reranker.rerank_with_cohere(query, candidates, top_n=k)


def show(label: str, results: list[tuple[str, float]]):
    print(label)
    for i, (doc_id, score) in enumerate(results[:5], 1):
        text = str(corpus_by_id.loc[doc_id, "text"])
        print(f"{i} [{score:.3f} {text[:70]}]")


if __name__ == "__main__":
    Query = "Where should I park my rainy-day funds?"
    print(Query)

    show("Hybrid", search_reranked(Query))
