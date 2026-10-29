import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import RASAS
from playlist import make_playlist, quotas


def stub(**p):
    def f(texts):
        v = np.full(len(RASAS), 1e-4)
        for k, x in p.items():
            v[RASAS.index(k)] = x
        return [v / v.sum()]
    return f


def test_quotas_sum_and_order():
    q = quotas([0.6, 0.3, 0.1], 10)
    assert sum(q) == 10 and q == sorted(q, reverse=True) and min(q) >= 1


def test_quotas_skewed_still_gives_each_one():
    q = quotas([0.96, 0.03, 0.01], 10)
    assert sum(q) == 10 and min(q) >= 1


def test_playlist_unique_and_spread():
    res = make_playlist("x", 10, scorer=stub(Shringara=0.6, Karuna=0.3, Veera=0.1))
    songs = res["songs"]
    assert len(songs) == 10
    assert len({s["metadata_path"] for s in songs}) == 10
    top = [s for s in songs if s["rasa"] == "Shringara"]
    assert len({s["raga"] for s in top}) >= 2
    counts = [sum(s["rasa"] == r for s in songs) for r in ("Shringara", "Karuna", "Veera")]
    assert counts == sorted(counts, reverse=True)


def test_shortfall_rolls_over():
    res = make_playlist("x", 12, scorer=stub(Shanta=0.8, Shringara=0.2))
    assert len(res["songs"]) == 12


def test_no_duplicate_titles():
    from playlist import dupe_key
    from raaga_search.base import Song
    mk = lambda t: Song(t, "c", [], [], "", Path(t))
    assert dupe_key(mk("Raag Yaman")) == dupe_key(mk("Raga Yaman"))
    assert dupe_key(mk("Idhu Thaano Thillai Sthalam")) == dupe_key(mk("Idhuthaano Thillai Sthalam"))
    for n in (10, 20):
        for mix in (dict(Shringara=0.6, Karuna=0.3, Veera=0.1), dict(Veera=0.5, Karuna=0.5)):
            songs = make_playlist("x", n, scorer=stub(**mix))["songs"]
            keys = [dupe_key(Song(s["title"], "", [], [], "", Path("p"))) for s in songs]
            assert len(keys) == len(set(keys))
