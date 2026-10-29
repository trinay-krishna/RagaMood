"""Usage: python run_all.py [vad|go|emb_max|emb_mean|all ...]

Each retriever's results land in results/<retriever>/ (see retriever_eval.run_evaluation).
"""
import sys

import pandas as pd

from common import RESULTS_DIR
from retriever_eval import run_evaluation
from sample_dataset import labeled

BHAY_BIB = [{"Bhayanaka", "Bibhatsa"}]


def _vad():
    from VADRetriever import vad_probs
    return run_evaluation(labeled, vad_probs, name="VAD only", equivalent=BHAY_BIB)

def _go():
    from GoRetriever import go_probs
    return run_evaluation(labeled, go_probs, name="GoEmotions")

def _emb(pool):
    def run():
        from embeddingRetriever import emb_probs, leakage_check
        leakage_check(labeled)
        return run_evaluation(labeled, lambda s: emb_probs(s, pool=pool), name=f"Embeddings (MiniLM, {pool})")
    return run

RUNNERS = {"vad": _vad, "go": _go, "emb_max": _emb("max"), "emb_mean": _emb("mean")}


if __name__ == "__main__":
    picks = [a for a in sys.argv[1:] if a != "all"] or list(RUNNERS)
    unknown = [p for p in picks if p not in RUNNERS]
    if unknown:
        sys.exit(f"unknown retriever(s) {unknown}; choose from {list(RUNNERS)} or 'all'")

    rows = {}
    for key in picks:
        rows[key] = {k: float(v) for k, v in RUNNERS[key]().items()}
    table = pd.DataFrame(rows).T
    print("\n", table.round(3))
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "comparison.csv")
