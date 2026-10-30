"""User text -> short ranked playlist: more songs from likelier rasas, spread over ragas/artists/forms.

Run:  python playlist.py "I miss my mother" [--n 10] [--top-k 3] [--temperature 0.5]
                         [--max-per-artist 2] [--seed S] [--json]     (no text: interactive)
Use:  from playlist import make_playlist
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from pipeline import DATA_ROOT, TRADITIONS, _scorer, text_to_rasas
from rasa_raga_map import RASA_TO_RAGAS, search_names, set_equivalents
from raaga_search import find_songs
from raaga_search.search import normalize
from raga_time import MATCH_PICKS, parse_time, time_match


def quotas(probs, n, temperature=0.5):
    """Split n slots over rasas by p**temperature (largest remainder); every rasa gets >=1 if n allows."""
    w = [p ** temperature for p in probs]
    tot = sum(w)
    share = [n * x / tot for x in w]
    q = [int(s) for s in share]
    if n >= len(q):
        q = [max(1, x) for x in q]
    order = sorted(range(len(q)), key=lambda i: -(share[i] - int(share[i])))
    i = 0
    while sum(q) < n:
        q[order[i % len(q)]] += 1
        i += 1
    while sum(q) > n:  # the >=1 floor can overshoot: trim the least likely rasa with spare slots
        j = max((k for k in range(len(q)) if q[k] > 1), key=lambda k: -probs[k], default=None)
        if j is None:
            break
        q[j] -= 1
    return q


def dupe_key(song):
    """Same piece in another concert/performance (or 'Raag X' vs 'Raga X') gets the same key."""
    t = normalize(song.title)
    for p in ("raag", "raga"):
        if t.startswith(p) and len(t) > len(p):
            return t[len(p):]
    return t


def _songs_by_raga(rasa, root, traditions, hour=None):
    """{raga: songs}, time-matched ragas first."""
    out = {}
    for raga in sorted(RASA_TO_RAGAS[rasa], key=lambda r: not time_match(r, hour)):
        found, _ = find_songs(search_names(raga), root, traditions)
        songs = {s.metadata_path: s for ss in found.values() for s in ss}
        if songs:
            out[raga] = list(songs.values())
    return out


def _pick(songs, chosen, max_per_artist, rng):
    """Best song: fewest artist/concert/form/tradition repeats among already chosen songs."""
    def key(s):
        a = sum(1 for c in chosen if set(c.artists) & set(s.artists))
        if a >= max_per_artist:
            return None
        return (a, sum(c.concert == s.concert for c in chosen), sum(c.form == s.form for c in chosen),
                sum(c.tradition == s.tradition for c in chosen), rng.random())
    scored = [(k, s) for s in songs if (k := key(s)) is not None]
    return min(scored, key=lambda x: x[0])[1] if scored else None


def make_playlist(text, n=10, top_k=3, temperature=0.5, max_per_artist=2, seed=0,
                  root=DATA_ROOT, traditions=TRADITIONS, scorer=_scorer, exclude=(), hour=None):
    """exclude: dupe_keys of songs that must not appear (e.g. already chosen elsewhere).
    hour: time of day (float, default = now); ragas of that slot go first and get MATCH_PICKS per pass."""
    rng = random.Random(seed)
    rasas = text_to_rasas(text, top_k, scorer)
    total = sum(p for _, p in rasas) or 1.0
    probs = [p / total for _, p in rasas]
    plan = quotas(probs, n, temperature)
    pools = {r: _songs_by_raga(r, root, traditions, hour) for r, _ in rasas}
    chosen, items, used = [], [], set(exclude)
    short = 0
    for (rasa, p), q in zip(rasas, plan):
        want, short = q + short, 0
        pool = {k: list(v) for k, v in pools[rasa].items()}
        while want and pool:
            for raga in list(pool):  # round-robin: one song per raga per pass (time-matched ragas get more)
                matched = time_match(raga, hour)
                for _ in range(MATCH_PICKS if matched else 1):
                    if not want:
                        break
                    cand = [s for s in pool[raga] if dupe_key(s) not in used]
                    s = _pick(cand, chosen, max_per_artist, rng)
                    if s is None:
                        pool.pop(raga)
                        break
                    used.add(dupe_key(s))
                    chosen.append(s)
                    pool[raga].remove(s)
                    items.append({"rasa": rasa, "prob": round(p, 3), "raga": raga, "time_match": matched,
                                  **s.to_dict()})
                    want -= 1
        short = want  # unfilled slots roll down to the next rasa
    return {"text": text, "rasas": [{"rasa": r, "prob": round(p, 3), "quota": q}
                                   for (r, p), q in zip(rasas, plan)], "songs": items}


def show(res):
    print("Rasas: " + ", ".join(f"{d['rasa']} ({d['prob']:.2f}, {d['quota']} slots)" for d in res["rasas"]))
    for i, s in enumerate(res["songs"], 1):
        extra = " / ".join(x for x in (s["tradition"], s["form"]) if x)
        print(f"{i:2}. [{s['rasa']} {s['prob']:.2f}] {s['raga']}{' *' if s['time_match'] else ''}: {s['title']} | "
              f"{', '.join(s['artists'])} | {s['concert']} | {extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.5, help="p**T; lower = flatter split")
    ap.add_argument("--max-per-artist", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--equivalents", action="store_true",
                    help="also search unverified cross-name/tradition counterparts (rasa_raga_map.RAGA_EQUIVALENTS)")
    ap.add_argument("--time", help="HH:MM for the time-of-day raga boost (default: now); * marks boosted ragas")
    a = ap.parse_args()
    set_equivalents(a.equivalents)
    texts = [a.text] if a.text else iter(lambda: input("text> ").strip(), "")
    for t in texts:
        res = make_playlist(t, a.n, a.top_k, a.temperature, a.max_per_artist, a.seed, hour=parse_time(a.time))
        if a.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            show(res)


if __name__ == "__main__":
    main()
