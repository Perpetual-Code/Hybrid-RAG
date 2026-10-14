"""
The payoff NDCG10 across BM25,Dense,Hybrid(RRF) and Hybrid+Rerank, on the FiQA-2018 test set

Normalized Discounted Cummulative Gain at 10(NDCG@10) is the standard rtetrieval metric.It rewards putting relevant docs high in the top-10 and penalizes them low.
Perfect ranking =1.0

We evaluate on 50 randomly sampled test queries so all four methods stay with Cohere's free tier(10  calls/min , 1000/month)
Set RERANK_SAMPLE_SIZE=None to run all 648 queries

Public BEIR baselines for FiQA NDCG@10:
BM25                        ~24
text-embedding-3-small       ~31
+cross-encoder rerank        ~40+

The wide spread between BM25 and reranked hybird is the point: BM25 alone struggles on paraphrase-heavy questions , and every stage of the pipeline earns its keep.

More info: https://github.com/beir-cellar/beir/wiki/Leaderboard
"""

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.fusion import hybrid_search
from utils.reranker import ReRanker
from utils.retrievers import BM25Retriever, DenseRetriever, load_corpus

corpus = load_corpus()
corpus_by_id = corpus.set_index("_id")
bm25_obj = BM25Retriever()
dense_obj = DenseRetriever()
RERANK_SAMPLE = 50
SEED = 15
RERANK_MODEL = "rerank-v4.0-fast"
reranker = ReRanker(RERANK_MODEL)
# from utils.retrievers import BM25Retriever, DenseRetriever

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "fiqa"


# ----------------------------------------------------------------------
# INFO STEP 1 -> Load Queries and Qrels(Ground truth), pick the sample
# -----------------------------------------------------------------------
def ndcg_at_k(predicted_ids: list[str], relevant: dict[str, int], k: int = 10):
    # INFO Discounted Cumulative Gain
    dcg = sum(
        relevant.get(doc_id, 0) / math.log2(2 + rank)
        for rank, doc_id in enumerate(predicted_ids[:k])
    )

    # INFO Ideal Cumulative Gains
    ideal_rels = sorted(relevant.values(), reverse=True)
    idcg = sum(rel / math.log2(2 + rank) for rank, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


# NOTE -> Just a small test to understand the math and logic behind the calculation
# NOTE -> Test values for debugging making sure the implementation is correct and I got the right answer 1.84
predicted = ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10"]
relevant = {"d1": 1, "d3": 1, "d10": 1}  # relevant at ranks 1, 3, 10
print(ndcg_at_k(predicted, relevant, k=10))

# ideal_rels=([1,1,1])
# idcg=sum(rel/math.log2(rank+2) for rank,rel in enumerate(ideal_rels))
# print(idcg)

# idcg_1=sum(rel/math.log2(1+rank) for rank,rel in enumerate(ideal_rels,start=1))
# print(idcg_1)


# --------------------------------------------------------------
# INFO STEP 2 -> Load Queries and Qrels(Ground truth), pick the sample
# --------------------------------------------------------------
qrels_df = pd.read_parquet(DATA_DIR / "qrels.parquet")
queries_df = pd.read_parquet(DATA_DIR / "queries.parquet")

qrels: dict[str, dict[str, int]] = defaultdict(dict)

for _, row in qrels_df.iterrows():
    qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])

queries_with_qrels = queries_df[queries_df["_id"].astype(str).isin(qrels)].copy()
sample = queries_with_qrels.sample(n=RERANK_SAMPLE, random_state=SEED)
print(f"Evaluating on {len(sample)} queries sampled from {len(queries_with_qrels)}")


def hybrid_topk(query: str, candidate_k=50) -> list[str]:
    return [
        d
        for d, _ in hybrid_search(query, bm25=bm25_obj, dense=dense_obj, k=candidate_k)
    ]


def reranked_topk(query: str, candidate_k: int = 50, k: int = 10):
    candidates = hybrid_topk(query, candidate_k=candidate_k)
    return [
        d
        for d, _ in reranker.rerank_with_cohere(
            query, candidate_ids=candidates, corpus_by_id=corpus_by_id, top_n=k
        )
    ]

    # return [d for d,_ in reranker.rerank_with_cohere(query,candidates) ]


results: dict[str, list[float]] = defaultdict(list)
for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Evaluating"):
    query_id = row["_id"]
    query_text = row["text"]
    relevant = qrels[query_id]

    # print(row["_id"],row['text'])

    results["BM25"].append(
        ndcg_at_k([d for d, _ in bm25_obj.bm25_search(query_text, k=10)], relevant)
    )
    results["Dense"].append(
        ndcg_at_k([d for d, _ in dense_obj.dense_search(query_text, k=10)], relevant)
    )
    # results["Hybrid (RRF)"].append(ndcg_at_k(hybrid_topk(query_text)[:10], relevant))
    results["Hybrid (RRF)"].append(ndcg_at_k(hybrid_topk(query_text[:10]), relevant))
    results["Hybrid + Rerank"].append(ndcg_at_k(reranked_topk(query_text), relevant))

Query = "Where should I park my rainy-day funds?"
reranked_topk(Query)


if __name__ == "__main__":
    print(f"\nNDCG@10 x100 on FiQA ({len(sample)} sampled test queries)")
    print("-" * 42)
    for method, scores in results.items():
        print(f"  {method:<22} {np.mean(scores) * 100:.2f}")
