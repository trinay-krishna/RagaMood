"""Mixed queries: an explicit entity (raga/artist/title/form) AND an emotion.

"I'm feeling sad today, I want songs with the raga Ahibhairav"
  -> the named entity is a constraint: its songs come first;
  -> the emotion fills the rest with other songs of the same mood (rasa pipeline).

Routing guide for the agent:  entity only -> hybrid.py | emotion only -> playlist.py | both -> mixed.py

Run:  python mixed.py --emotion "I'm feeling sad today" --entity "raga Ahibhairav" [--n 10]
      python mixed.py "I'm feeling sad today, play raga Ahibhairav"      (same text used for both parts)
Use:  from mixed import mixed_search
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from hybrid import Index, tokens
from pipeline import DATA_ROOT, TRADITIONS, _scorer, text_to_rasas
from playlist import _pick, dupe_key, make_playlist
from raaga_search import find_songs
from raga_time import parse_time
from rasa_raga_map import RASA_TO_RAGAS, rasa_for_raga, search_names, set_equivalents

STOP = {"raga", "raag", "ragam", "song", "songs", "music", "by", "in", "the", "a", "an", "of", "to", "with",
        "listen", "play", "find", "some", "me", "i", "want", "like", "from", "and", "for", "on", "my"}


MATCH_MIN_PROB = 0.05  # a rasa only counts as "the user's mood" above this probability


def _grams(text):
    t = tokens(text)
    return set(t) | {a + b for a, b in zip(t, t[1:])}


def resolve_ragas(entity):
    """Mapped ragas whose name (or Saraga spelling) appears in the entity text."""
    grams = _grams(entity)
    return [raga for ragas in RASA_TO_RAGAS.values() for raga in ragas
            if any("".join(tokens(n)) in grams for n in search_names(raga) if len("".join(tokens(n))) >= 3)]


def entity_songs(entity, index, root=DATA_ROOT, traditions=TRADITIONS):
    """[(label, song)] for the entity: mapped ragas first (alias-aware), then confident BM25 matches."""
    out, seen = [], set()
    ragas = resolve_ragas(entity)
    for raga in ragas:
        found, _ = find_songs(search_names(raga), root, traditions)
        for s in {s.metadata_path: s for ss in found.values() for s in ss}.values():
            if s.metadata_path not in seen:
                seen.add(s.metadata_path)
                out.append((raga, s))
    if not ragas:  # artist / title / form / unmapped raga: keep only songs whose fields share a query token
        q = [t for t in tokens(entity) if t not in STOP]
        scores = index.sparse(" ".join(q))
        for i in sorted(range(len(scores)), key=lambda i: -scores[i]):
            s = index.songs[i]
            if scores[i] <= 0:
                break
            fields = set(tokens(" ".join([s.title, *s.raagas, *s.artists, s.form, s.taala, s.concert])))
            if q and fields & set(q) and s.metadata_path not in seen:
                seen.add(s.metadata_path)
                out.append((entity.strip(), s))
    return ragas, out


def mixed_search(emotion, entity, n=10, requested_share=0.6, top_k=3, max_per_artist=2, seed=0,
                 index=None, scorer=_scorer, root=DATA_ROOT, traditions=TRADITIONS, hour=None):
    ix = index or Index(root, traditions)
    rng = random.Random(seed)
    rasas = text_to_rasas(emotion, top_k, scorer)
    ragas, pool = entity_songs(entity, ix, root, traditions)

    # consistency: does the requested raga's rasa agree with the emotion?
    raga_rasas = sorted({rasa_for_raga(r) for r in ragas})
    top = {r for r, p in rasas if p >= MATCH_MIN_PROB}
    fit = "n/a" if not raga_rasas else ("match" if top & set(raga_rasas) else "mismatch")

    want = round(n * requested_share)
    chosen, items, used = [], [], set()
    for _ in range(want):
        cand = [(lab, s) for lab, s in pool if dupe_key(s) not in used]
        s = _pick([s for _, s in cand], chosen, max_per_artist, rng)
        if s is None:
            break
        lab = next(lab for lab, x in cand if x is s)
        used.add(dupe_key(s))
        chosen.append(s)
        items.append({"section": "requested", "label": lab, **s.to_dict()})

    block = {dupe_key(s) for _, s in pool}  # mood section adds *other* songs, not more of the entity
    rest = n - len(items)
    mood = make_playlist(emotion, rest, top_k, max_per_artist=max_per_artist, seed=seed, root=root,
                         traditions=traditions, scorer=scorer, exclude=used | block, hour=hour)["songs"] if rest > 0 else []
    for s in mood:
        items.append({"section": "mood", "label": f"{s['rasa']} {s['prob']:.2f}", **s})
    return {"emotion": emotion, "entity": entity, "rasas": [{"rasa": r, "prob": round(p, 3)} for r, p in rasas],
            "entity_ragas": ragas, "entity_rasas": raga_rasas, "fit": fit,
            "entity_matches": len(pool), "songs": items}


def show(res):
    print("Rasas: " + ", ".join(f"{d['rasa']} ({d['prob']:.2f})" for d in res["rasas"]))
    if res["entity_matches"] == 0:
        print(f"Entity '{res['entity']}': no songs found in the dataset; showing mood songs only.")
    else:
        extra = f" (rasa {', '.join(res['entity_rasas'])}: {res['fit']} with your mood)" if res["entity_rasas"] else ""
        print(f"Entity '{res['entity']}': {res['entity_matches']} song(s) in the dataset{extra}")
    for i, s in enumerate(res["songs"], 1):
        tag = f"Requested: {s['label']}" if s["section"] == "requested" else f"Same mood: {s['label']}"
        extra = " / ".join(x for x in (s["tradition"], s["form"]) if x)
        raga = s.get("raga") or ", ".join(dict.fromkeys(s["raagas"]))
        print(f"{i:2}. [{tag}] {raga}: {s['title']} | {', '.join(s['artists'])} | {s['concert']} | {extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--emotion")
    ap.add_argument("--entity")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--requested-share", type=float, default=0.6)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-per-artist", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--equivalents", action="store_true",
                    help="also search unverified cross-name/tradition counterparts (rasa_raga_map.RAGA_EQUIVALENTS)")
    ap.add_argument("--time", help="HH:MM for the time-of-day raga boost in the mood section (default: now)")
    a = ap.parse_args()
    set_equivalents(a.equivalents)
    emotion, entity = a.emotion or a.text, a.entity or a.text
    if not emotion or not entity:
        ap.error("give a text, or both --emotion and --entity")
    res = mixed_search(emotion, entity, a.n, a.requested_share, a.top_k, a.max_per_artist, a.seed,
                       hour=parse_time(a.time))
    print(json.dumps(res, ensure_ascii=False, indent=2) if a.json else "", end="")
    if not a.json:
        show(res)


if __name__ == "__main__":
    main()
