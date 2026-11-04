"""Harder known-item / attribute search evaluation (no LLM).  Run: python evaluation/run_hard_search_eval.py
-> evaluation/results/hard_search_metrics.json

The easy test in run_retrieval_eval.py queries with the exact title, which is text the index already holds.
Here the queries are degraded or underspecified:
  target queries (one correct piece, found by title key): partial title, two typos, spelling variant,
                                                         artist + raga (no title), form + raga + artist
  set queries (many correct songs; binary relevance, nDCG against the catalogue count): raga only,
                                                         artist + form, tradition + form + raga
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from hybrid import Index, search
from raaga_search.search import normalize
from rasa_raga_map import search_names
from run_retrieval_eval import SEED, key_of, ndcg, typo
from playlist import dupe_key

OUT = HERE / "results"
MODES = ("sparse", "dense", "hybrid")
VARIANTS = [("aa", "a"), ("ee", "i"), ("oo", "u"), ("th", "t"), ("dh", "d"), ("sh", "s"), ("v", "w"), ("kh", "k")]


def variant(text):
    """Common transliteration drift: 'Thillana' -> 'Tilana', 'Shankarabharanam' -> 'Sankarabharanam'."""
    t = text.lower()
    for a, b in VARIANTS:
        t = t.replace(a, b)
    return t


def partial(title, rng):
    """Half of the words of a multi-word title (the longest word for a 2-word title); None if too short."""
    words = title.split()
    if len(words) < 2:
        return None
    if len(words) == 2:
        return max(words, key=len)
    k = max(1, len(words) // 2)
    start = rng.randrange(0, len(words) - k + 1)
    return " ".join(words[start:start + k])


def raga_names(raga):
    return {normalize(n) for n in search_names(raga)} | {normalize(raga)}


def has_raga(song, names):
    return any(normalize(r) in names for r in (song.get("raagas") or []))


def target_query_sets(ix, rng, n):
    songs = [s for s in ix.songs if len(s.title) >= 4 and s.artists and s.raagas]
    sample = rng.sample(songs, min(n, len(songs)))
    sets = {"partial title": [], "two typos": [], "spelling variant": [],
            "artist + raga": [], "form + raga + artist": []}
    for s in sample:
        p = partial(s.title, rng)
        if p:
            sets["partial title"].append((s, p))
        sets["two typos"].append((s, typo(typo(s.title, rng), rng)))
        sets["spelling variant"].append((s, variant(s.title)))
        sets["artist + raga"].append((s, f"{s.artists[0]} {s.raagas[0]}"))
        if s.form:
            sets["form + raga + artist"].append((s, f"{s.form} {s.raagas[0]} {s.artists[0]}"))
    return sets


def run_target(ix, name, pairs):
    rows = []
    for mode in MODES:
        ranks, prec = [], []
        for s, q in pairs:
            res = search(q, ix, n=10, mode=mode)["songs"]
            key = dupe_key(s)
            ranks.append(next((i for i, r in enumerate(res, 1) if key_of(r["title"]) == key), None))
        r = np.array([x or 0 for x in ranks])
        rows.append({"mode": mode, "query": name, "n": len(ranks),
                     "recall@1": float(np.mean(r == 1)), "recall@5": float(np.mean((r >= 1) & (r <= 5))),
                     "recall@10": float(np.mean(r >= 1)),
                     "mrr": float(np.mean([1 / x if x else 0 for x in ranks])),
                     "ndcg@10": float(np.mean([1 / np.log2(x + 1) if x else 0 for x in ranks]))})
    return rows


def set_queries(ix, rng, n):
    """(name, query, predicate(song dict) -> bool) with relevance defined by the metadata the query names."""
    out = {"raga only": [], "artist + form": [], "tradition + form + raga": []}
    ragas = sorted({s.raagas[0] for s in ix.songs if s.raagas})
    for r in rng.sample(ragas, min(n, len(ragas))):
        names = raga_names(r)
        out["raga only"].append((f"raga {r}", lambda s, nm=names: has_raga(s, nm)))
    combos = sorted({(s.artists[0], s.form) for s in ix.songs if s.artists and s.form})
    rng.shuffle(combos)
    for a, f in combos[:n]:
        out["artist + form"].append((f"{a} {f}", lambda s, a=a, f=f: a in (s.get("artists") or []) and s.get("form") == f))
    trip = sorted({(s.tradition, s.form, s.raagas[0]) for s in ix.songs if s.form and s.raagas and s.tradition})
    rng.shuffle(trip)
    for t, f, r in trip[:n]:
        names = raga_names(r)
        out["tradition + form + raga"].append(
            (f"{t} {f} in {r}", lambda s, t=t, f=f, nm=names: s.get("tradition") == t and s.get("form") == f and has_raga(s, nm)))
    return out


def run_set(ix, name, queries):
    songs = [s.to_dict() for s in ix.songs]
    rows = []
    for mode in MODES:
        p, nd, top1 = [], [], []
        for q, pred in queries:
            n_rel = sum(bool(pred(s)) for s in songs)
            if not n_rel:
                continue
            res = search(q, ix, n=10, mode=mode)["songs"]
            rel = [int(bool(pred(s))) for s in res]
            p.append(np.mean(rel) if rel else 0.0)
            nd.append(ndcg(rel, n_rel))
            top1.append(rel[0] if rel else 0)
        rows.append({"mode": mode, "query": name, "n": len(p), "precision@10": float(np.mean(p)),
                     "ndcg@10": float(np.mean(nd)), "top1_relevant": float(np.mean(top1))})
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    ix = Index()
    rng = random.Random(SEED)
    print(f"{len(ix.songs)} songs")
    target_rows = []
    for name, pairs in target_query_sets(ix, rng, 300).items():
        target_rows += run_target(ix, name, pairs)
        print("done", name, len(pairs))
    set_rows = []
    for name, qs in set_queries(ix, rng, 60).items():
        set_rows += run_set(ix, name, qs)
        print("done", name, len(qs))
    (OUT / "hard_search_metrics.json").write_text(
        json.dumps({"target": target_rows, "set": set_rows}, indent=2), encoding="utf-8")
    import pandas as pd
    print(pd.DataFrame(target_rows).round(3).to_string(index=False))
    print(pd.DataFrame(set_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
