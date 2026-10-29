import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import RASAS
from hybrid import Index
from mixed import mixed_search, resolve_ragas
from playlist import dupe_key


def stub(**p):
    def f(texts):
        v = np.full(len(RASAS), 1e-4)
        for k, x in p.items():
            v[RASAS.index(k)] = x
        return [v / v.sum()]
    return f


@pytest.fixture(scope="module")
def ix():
    return Index()


def test_resolve_ragas():
    assert resolve_ragas("songs with the raga Ahibhairav") == ["Ahibhairav"]
    assert "Kalyani" in resolve_ragas("kalyani")


def test_requested_first_and_match(ix):
    r = mixed_search("sad", "raga Ahibhairav", 10, index=ix, scorer=stub(Karuna=0.9, Shanta=0.1))
    s = r["songs"]
    assert r["fit"] == "match" and len(s) == 10
    assert all(x["section"] == "requested" for x in s[:2]) and s[0]["label"] == "Ahibhairav"
    assert [x["section"] for x in s] == sorted((x["section"] for x in s), reverse=True)  # requested, then mood
    keys = [dupe_key(type("S", (), {"title": x["title"]})) for x in s]
    assert len(keys) == len(set(keys))


def test_mismatch_flagged_but_raga_kept(ix):
    r = mixed_search("happy", "Ahibhairav", 10, index=ix, scorer=stub(Shringara=0.9, Hasya=0.1))
    assert r["fit"] == "mismatch" and r["songs"][0]["section"] == "requested"


def test_zero_song_raga_falls_back_to_mood(ix):
    r = mixed_search("sad", "Madhuvanti", 8, index=ix, scorer=stub(Karuna=0.9))
    assert r["entity_matches"] == 0 and len(r["songs"]) == 8
    assert all(x["section"] == "mood" for x in r["songs"])


def test_artist_entity(ix):
    r = mixed_search("calm", "Sanjay Subrahmanyan", 6, index=ix, scorer=stub(Shanta=0.9))
    assert r["fit"] == "n/a" and r["songs"][0]["section"] == "requested"


def test_ahibhairav_strict_has_no_ahiri(ix):
    r = mixed_search("sad", "Ahibhairav", 10, index=ix, scorer=stub(Karuna=0.9))
    req = [s for s in r["songs"] if s["section"] == "requested"]
    assert req and not any("ahiri" in "".join(s["raagas"]).lower() for s in req)
