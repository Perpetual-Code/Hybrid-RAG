from collections import defaultdict

SMOOTHING_FACTOR_K = 60


def rrf(rankings: list[list[str]], k: int = SMOOTHING_FACTOR_K):
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
