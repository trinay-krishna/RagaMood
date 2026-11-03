"""Offline retrieval evaluation: known-item search, mood playlists, mixed queries, time-of-day boost.

Run:  python evaluation/run_retrieval_eval.py        -> evaluation/results/retrieval_metrics.json
No LLM involved. Mood-playlist "precision" uses the paper's rasa->raga map, i.e. it measures end-to-end
consistency with that map, not independent musical ground truth.
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import playlist as playlist_mod
from hybrid import Index, search
from mixed import mixed_search
from pipeline import _scorer
from playlist import dupe_key, make_playlist
from raaga_search.search import normalize
from raga_time import time_match
from rasa_raga_map import RASA_TO_RAGAS, search_names

OUT = HERE / "results"
SEED = 0
MAP_RASAS = list(RASA_TO_RAGAS)                      # the rasas that have ragas


def rasa_names(rasa):
    return {normalize(n) for raga in RASA_TO_RAGAS[rasa] for n in search_names(raga)}


def in_rasa(song, rasa):
    names = rasa_names(rasa)
    return any(normalize(r) in names for r in (song.get("raagas") or []))


def ndcg(rel, n_relevant, k=10):
    """nDCG@k for binary relevance; the ideal list has min(k, n_relevant) relevant items."""
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel[:k]))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, n_relevant)))
    return float(dcg / ideal) if ideal else 0.0


def memo_scorer():
    cache = {}

    def f(texts):
        miss = [t for t in texts if t not in cache]
        if miss:
            for t, p in zip(miss, _scorer(miss)):
                cache[t] = p
        return [cache[t] for t in texts]
    return f


class _Title:
    """dupe_key only needs .title"""

    def __init__(self, title):
        self.title = title


def key_of(title):
    return dupe_key(_Title(title))


def typo(text, rng):
    """Drop or swap one letter in a longer word (simulates a misspelling)."""
    words = text.split()
    idx = [i for i, w in enumerate(words) if len(w) >= 5]
    if not idx:
        return text
    i = rng.choice(idx)
    w = words[i]
    j = rng.randrange(1, len(w) - 1)
    words[i] = w[:j] + w[j + 1:] if rng.random() < 0.5 else w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


# ---------------------------------------------------------------- 1. known-item search
def known_item(ix, n_songs=300):
    rng = random.Random(SEED)
    songs = [s for s in ix.songs if len(s.title) >= 4 and s.artists]
    sample = rng.sample(songs, min(n_songs, len(songs)))
    qtypes = {"title": lambda s: s.title,
              "title+artist": lambda s: f"{s.title} {s.artists[0]}",
              "title (typo)": lambda s: typo(s.title, rng)}
    rows = []
    for mode in ("sparse", "dense", "hybrid"):
        for qname, mk in qtypes.items():
            ranks = []
            for s in sample:
                res = search(mk(s), ix, n=10, mode=mode)["songs"]
                key = dupe_key(s)
                pos = next((i for i, r in enumerate(res, 1) if key_of(r["title"]) == key), None)
                ranks.append(pos)
            r = np.array([x or 0 for x in ranks])
            rows.append({"mode": mode, "query": qname, "n": len(ranks),
                         "recall@1": float(np.mean(r == 1)), "recall@5": float(np.mean((r >= 1) & (r <= 5))),
                         "recall@10": float(np.mean(r >= 1)),
                         "mrr": float(np.mean([1 / x if x else 0 for x in ranks])),
                         "ndcg@10": float(np.mean([1 / np.log2(x + 1) if x else 0 for x in ranks]))})
    # attribute queries: artist + form, relevance = result matches both
    attr = []
    combos = sorted({(s.artists[0], s.form) for s in ix.songs if s.artists and s.form})
    rng.shuffle(combos)
    for mode in ("sparse", "dense", "hybrid"):
        prec = []
        for artist, form in combos[:60]:
            res = search(f"{artist} {form}", ix, n=10, mode=mode)["songs"]
            if res:
                prec.append(np.mean([artist in r["artists"] and form == r["form"] for r in res]))
            else:
                prec.append(0.0)
        attr.append({"mode": mode, "query": "artist+form", "n": len(prec), "precision@10": float(np.mean(prec))})
    return rows, attr


# ---------------------------------------------------------------- 2. mood playlists
def mood_playlists(ix, per_rasa=20):
    df = pd.read_csv(ROOT / "rasa_dataset.csv", encoding="utf-8-sig", keep_default_na=False)
    rng = random.Random(SEED)
    scorer = memo_scorer()
    songs = [s.to_dict() for s in ix.songs]
    out = {"playlist": [], "dense": [], "random": []}
    nd_all = {"playlist": [], "dense": [], "random": []}
    div, empty, total = [], 0, 0
    per_rasa_rows = []
    for rasa in MAP_RASAS:
        sents = df[df["primary_rasa"].str.strip() == rasa]["user_text"].tolist()
        sents = rng.sample(sents, min(per_rasa, len(sents)))
        got = {"playlist": [], "dense": [], "random": []}
        nd = {"playlist": [], "dense": [], "random": []}
        n_rel = sum(in_rasa(s, rasa) for s in songs)
        for t in sents:
            total += 1
            pl = make_playlist(t, n=10, scorer=scorer)["songs"]
            if not pl:
                empty += 1
            got["playlist"].append(np.mean([in_rasa(s, rasa) for s in pl]) if pl else 0.0)
            nd["playlist"].append(ndcg([int(in_rasa(s, rasa)) for s in pl], n_rel))
            if pl:
                div.append({"artists": len({a for s in pl for a in s["artists"]}),
                            "ragas": len({s["raga"] for s in pl}), "forms": len({s["form"] for s in pl if s["form"]}),
                            "dupes": len(pl) - len({key_of(s["title"]) for s in pl}),
                            "size": len(pl)})
            dn = search(t, ix, n=10, mode="dense")["songs"]
            got["dense"].append(np.mean([in_rasa(s, rasa) for s in dn]) if dn else 0.0)
            rnd = rng.sample(songs, 10)
            got["random"].append(np.mean([in_rasa(s, rasa) for s in rnd]))
            nd["dense"].append(ndcg([int(in_rasa(s, rasa)) for s in dn], n_rel))
            nd["random"].append(ndcg([int(in_rasa(s, rasa)) for s in rnd], n_rel))
        for k in got:
            out[k] += got[k]
            nd_all[k] += nd[k]
        per_rasa_rows.append({"rasa": rasa, **{k: float(np.mean(v)) for k, v in got.items()},
                              **{f"ndcg_{k}": float(np.mean(v)) for k, v in nd.items()}})
    d = pd.DataFrame(div)
    return {"precision@10": {k: float(np.mean(v)) for k, v in out.items()},
            "ndcg@10": {k: float(np.mean(v)) for k, v in nd_all.items()},
            "per_rasa": per_rasa_rows, "empty_rate": empty / total, "n_queries": total,
            "diversity": d.mean().round(2).to_dict()}


# ---------------------------------------------------------------- 3. mixed queries
def mixed(ix):
    df = pd.read_csv(ROOT / "rasa_dataset.csv", encoding="utf-8-sig", keep_default_na=False)
    rng = random.Random(SEED)
    scorer = memo_scorer()
    ents = ["Kalyani", "Bageshri", "Mohana", "Kamas", "Behag", "Shankarabharanam", "Ahibhairav"]
    moods = ["Karuna", "Hasya", "Shringara", "Shanta", "Veera"]
    req_share, req_hit, mood_prec = [], [], []
    for ent in ents:
        for rasa in moods:
            t = rng.choice(df[df["primary_rasa"].str.strip() == rasa]["user_text"].tolist())
            res = mixed_search(t, f"raga {ent}", n=10, index=ix, scorer=scorer)
            req = [s for s in res["songs"] if s["section"] == "requested"]
            mood = [s for s in res["songs"] if s["section"] == "mood"]
            req_share.append(len(req) / max(1, len(res["songs"])))
            names = {normalize(n) for n in search_names(ent)}
            if req:
                req_hit.append(np.mean([any(normalize(r) in names for r in s.get("raagas") or []) for s in req]))
            if mood:
                mood_prec.append(np.mean([in_rasa(s, rasa) for s in mood]))
    return {"n_queries": len(ents) * len(moods), "requested_share": float(np.mean(req_share)),
            "requested_entity_hit": float(np.mean(req_hit)) if req_hit else None,
            "mood_section_precision@rasa": float(np.mean(mood_prec)) if mood_prec else None}


# ---------------------------------------------------------------- 4. time-of-day boost
def time_boost(per_rasa=6):
    df = pd.read_csv(ROOT / "rasa_dataset.csv", encoding="utf-8-sig", keep_default_na=False)
    rng = random.Random(SEED)
    scorer = memo_scorer()
    sents = [t for r in ("Karuna", "Shringara", "Hasya", "Veera")
             for t in rng.sample(df[df["primary_rasa"].str.strip() == r]["user_text"].tolist(), per_rasa)]
    real = playlist_mod.time_match
    rows = []
    for hour in (5, 14, 17, 20, 23):
        share = {}
        for label, fn in (("boost", real), ("no boost", lambda *a, **k: False)):
            playlist_mod.time_match = fn
            vals = []
            for t in sents:
                pl = make_playlist(t, n=10, scorer=scorer, hour=hour)["songs"]
                if pl:
                    vals.append(np.mean([real(s["raga"], hour) for s in pl]))
            share[label] = float(np.mean(vals))
        rows.append({"hour": hour, **share})
    playlist_mod.time_match = real
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    ix = Index()
    print(f"{len(ix.songs)} songs")
    res = {}
    res["known_item"], res["attribute"] = known_item(ix)
    print("known-item done")
    res["mood_playlists"] = mood_playlists(ix)
    print("mood playlists:", res["mood_playlists"]["precision@10"])
    res["mixed"] = mixed(ix)
    print("mixed:", res["mixed"])
    res["time_boost"] = time_boost()
    print("time boost:", res["time_boost"])
    (OUT / "retrieval_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
