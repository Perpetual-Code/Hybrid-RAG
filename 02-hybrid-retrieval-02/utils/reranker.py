import os

import cohere
import pandas as pd
from dotenv import load_dotenv

# from utils.fusion import hybrid_search
from utils.retrievers import BM25Retriever, DenseRetriever

load_dotenv()
bm25_obj = BM25Retriever()
dense_obj = DenseRetriever()


class ReRanker:
    def __init__(self, ReRankModel) -> None:
        self.co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
        self.model = ReRankModel

    # INFO Call the Hybrid Search Function and get the IDs, which is used to call Rerank function
    def search_reranked(self, query: str, corpus_by_id: pd.DataFrame, k: int = 10):
        # hybrid_candidates = hybrid_search(query, bm25_obj, dense_obj)
        pass

    def rerank_with_cohere(
        self,
        query: str,
        candidate_ids: list[str],
        corpus_by_id: pd.DataFrame,
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        candidate_texts = [str(corpus_by_id.loc[d, "text"]) for d in candidate_ids]

        response = self.co.rerank(
            model=self.model, query=query, documents=candidate_texts, top_n=top_n
        )

        return [(candidate_ids[r.index], r.relevance_score) for r in response.results]
