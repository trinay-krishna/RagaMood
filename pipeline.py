"""User text -> rasas -> ragas -> Saraga songs.

Run:  python pipeline.py "I miss my mother" [--top-k 3] [--json]     (no text: interactive)
Use:  from pipeline import songs_for_text
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rasa_compare"))
sys.stdout.reconfigure(encoding="utf-8")

from common import RASAS
from rasa_raga_map import RASA_TO_RAGAS, search_names, set_equivalents
from raaga_search import find_songs

DATA_ROOT = ROOT / "data" / "saraga"
TRADITIONS = ("carnatic", "hindustani")


def _scorer(texts):
    # the cached models were pickled from scripts run as __main__; expose their classes there
    import __main__
    from rerank import rerank as rr
    from vad_supervised import vad_supervised as vs
    for mod in (rr, vs):
        for name in ("Reranker", "Model"):
            if hasattr(mod, name):
                setattr(__main__, name, getattr(mod, name))
    from rerank.rerank import rerank_probs
    return rerank_probs(texts)


def text_to_rasas(text, top_k=3, scorer=_scorer):
    """Top-k rasas with ragas, as [(rasa, prob)]; unmapped rasas are skipped, best mapped one is the fallback."""
    probs = scorer([text])[0]
    ranked = sorted(zip(RASAS, map(float, probs)), key=lambda x: -x[1])
    mapped = [(r, p) for r, p in ranked if r in RASA_TO_RAGAS]
    top = [(r, p) for r, p in ranked[:top_k] if r in RASA_TO_RAGAS]
    return top or mapped[:1]


def rasas_to_ragas(rasas):
    """{rasa: [raga, ...]} for the given rasas."""
    return {r: RASA_TO_RAGAS[r] for r in rasas}


def songs_for_text(text, top_k=3, root=DATA_ROOT, traditions=TRADITIONS, scorer=_scorer):
    rasas = text_to_rasas(text, top_k, scorer)
    out = {"text": text, "rasas": [{"rasa": r, "prob": round(p, 3)} for r, p in rasas], "ragas": []}
    for rasa, ragas in rasas_to_ragas([r for r, _ in rasas]).items():
        for raga in ragas:
            found, _ = find_songs(search_names(raga), root, traditions)
            songs = {s.metadata_path: s for ss in found.values() for s in ss}.values()
            out["ragas"].append({"rasa": rasa, "raga": raga,
                                 "songs": [s.to_dict() for s in songs]})
    return out


def show(res):
    print("Rasas: " + ", ".join(f"{d['rasa']} ({d['prob']:.2f})" for d in res["rasas"]))
    for e in res["ragas"]:
        print(f"\n[{e['rasa']}] {e['raga']}: {len(e['songs'])} song(s)")
        for s in e["songs"]:
            print(f"  - {s['title']} | {', '.join(s['artists'])} | {s['concert']}")
    none = [e["raga"] for e in res["ragas"] if not e["songs"]]
    if none:
        print("\nNo songs in dataset for: " + ", ".join(none))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--equivalents", action="store_true",
                    help="also search unverified cross-name/tradition counterparts (rasa_raga_map.RAGA_EQUIVALENTS)")
    a = ap.parse_args()
    set_equivalents(a.equivalents)
    texts = [a.text] if a.text else iter(lambda: input("text> ").strip(), "")
    for t in texts:
        res = songs_for_text(t, a.top_k)
        print(json.dumps(res, ensure_ascii=False, indent=2) if a.json else "", end="")
        if not a.json:
            show(res)


if __name__ == "__main__":
    main()
