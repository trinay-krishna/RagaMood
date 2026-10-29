"""Hybrid song search over the Saraga metadata: BM25 (sparse) + MiniLM (dense), fused with RRF.

Independent of the rasa/raga mapping: the query is matched straight against each song's metadata.
Sparse catches exact names (raga, artist, form, taala, title words); dense catches paraphrases,
misspellings and descriptive queries ("slow classical vocal piece").

Run:  python hybrid.py "Sanjay Subrahmanyan kriti in Kalyani" [--n 10] [--mode hybrid|dense|sparse]
                       [--pool 50] [--w-dense 1] [--w-sparse 1] [--max-per-artist 2] [--json]
Use:  from hybrid import search
"""
import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from raaga_search.registry import LOADERS
from pipeline import DATA_ROOT, TRADITIONS
from playlist import dupe_key

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = ROOT / "cache"
RRF_K = 60

# Generic music-form glosses (not rasa labels), appended to documents so descriptive queries can match.
GLOSS = {
    "vilambit": "slow", "madhya": "medium tempo", "drut": "fast",
    "thillana": "fast rhythmic dance piece", "tarana": "fast rhythmic syllables",
    "varnam": "technical practice piece", "kriti": "composed devotional song",
    "khayal": "classical vocal improvisation", "alaap": "slow unmetered exploration",
    "mangalam": "auspicious closing piece", "bhajan": "devotional song", "viruttham": "devotional verse",
    "shloka": "devotional verse recitation", "javali": "light romantic song",
    "thumri": "light romantic semi-classical", "dadra": "light semi-classical",
    "pallavi": "rhythmic improvisation", "ragamalika": "garland of many ragas",
}


def tokens(text):
    s = unicodedata.normalize("NFKD", text)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.findall(r"[a-z0-9]+", s)


def doc_text(song):
    meta = [song.tradition, song.form, song.taala, song.laya]
    gloss = [GLOSS[t] for field in (song.form, song.laya) for t in tokens(field) if t in GLOSS]
    return " . ".join(x for x in [song.title, ", ".join(song.raagas), *meta,
                                  ", ".join(song.artists), song.concert, *gloss] if x)


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        # title and raga text are repeated once as a cheap field boost
        self.docs = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = self.len.mean() or 1.0
        self.k1, self.b = k1, b
        df = Counter(t for d in self.docs for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query_tokens):
        out = np.zeros(len(self.docs))
        for t in set(query_tokens):
            if t not in self.idf:
                continue
            for i, d in enumerate(self.docs):
                f = d.get(t, 0)
                if f:
                    out[i] += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


def rrf(rankings, weights, k=RRF_K):
    """rankings: list of ranked index lists -> {index: fused score}."""
    fused = {}
    for ranking, w in zip(rankings, weights):
        for r, i in enumerate(ranking, 1):
            fused[i] = fused.get(i, 0.0) + w / (k + r)
    return fused


class Index:
    def __init__(self, root=DATA_ROOT, traditions=TRADITIONS):
        self.songs = [s for t in traditions for s in LOADERS[t](root).load_songs()]
        self.texts = [doc_text(s) for s in self.songs]
        self.bm25 = BM25([tokens(s.title) + tokens(", ".join(s.raagas)) + tokens(t)
                          for s, t in zip(self.songs, self.texts)])
        self._enc = None
        self._emb = None

    def _encoder(self):
        if self._enc is None:
            from sentence_transformers import SentenceTransformer
            self._enc = SentenceTransformer(MODEL)
        return self._enc

    @property
    def emb(self):
        if self._emb is None:
            h = hashlib.md5("\n".join(self.texts).encode("utf8")).hexdigest()[:12]
            f = CACHE / f"hybrid_dense_{MODEL.split('/')[-1]}_{h}.npy"
            if f.exists():
                self._emb = np.load(f)
            else:
                self._emb = self._encoder().encode(self.texts, normalize_embeddings=True,
                                                   batch_size=64, show_progress_bar=False)
                CACHE.mkdir(exist_ok=True)
                np.save(f, self._emb)
        return self._emb

    def sparse(self, q):
        s = self.bm25.scores(tokens(q))
        return s

    def dense(self, q):
        v = self._encoder().encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
        return self.emb @ v


def _ranked(scores, pool, require_positive=False):
    order = np.argsort(-scores, kind="stable")[:pool]
    return [int(i) for i in order if not require_positive or scores[i] > 0]


def search(q, index=None, n=10, mode="hybrid", pool=50, w_dense=1.0, w_sparse=1.0, max_per_artist=2):
    ix = index or Index()
    d_rank = _ranked(ix.dense(q), pool) if mode in ("hybrid", "dense") else []
    s_rank = _ranked(ix.sparse(q), pool, True) if mode in ("hybrid", "sparse") else []
    fused = rrf([d_rank, s_rank], [w_dense, w_sparse])
    d_pos = {i: r for r, i in enumerate(d_rank, 1)}
    s_pos = {i: r for r, i in enumerate(s_rank, 1)}
    items, seen, artists = [], set(), Counter()
    for i in sorted(fused, key=lambda i: -fused[i]):
        s = ix.songs[i]
        key = dupe_key(s)
        names = set(s.artists)
        if key in seen or any(artists[a] >= max_per_artist for a in names):
            continue
        seen.add(key)
        artists.update(names)
        items.append({"score": round(fused[i], 5), "dense_rank": d_pos.get(i), "sparse_rank": s_pos.get(i),
                      "raga": ", ".join({"".join(tokens(r)): r for r in reversed(s.raagas)}.values()),
                      **s.to_dict()})
        if len(items) == n:
            break
    return {"query": q, "mode": mode, "songs": items}


def show(res):
    print(f"Query: {res['query']}  (mode: {res['mode']})")
    if not res["songs"]:
        print("No matches.")
    for i, s in enumerate(res["songs"], 1):
        extra = " / ".join(x for x in (s["tradition"], s["form"]) if x)
        ranks = f"D#{s['dense_rank'] or '-'} S#{s['sparse_rank'] or '-'}"
        print(f"{i:2}. [{s['score']:.4f} {ranks}] {s['raga'] or '?'}: {s['title']} | "
              f"{', '.join(s['artists'])} | {s['concert']} | {extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--w-dense", type=float, default=1.0)
    ap.add_argument("--w-sparse", type=float, default=1.0)
    ap.add_argument("--max-per-artist", type=int, default=2)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    ix = Index()
    texts = [a.text] if a.text else iter(lambda: input("query> ").strip(), "")
    for t in texts:
        res = search(t, ix, a.n, a.mode, a.pool, a.w_dense, a.w_sparse, a.max_per_artist)
        if a.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            show(res)


if __name__ == "__main__":
    main()
