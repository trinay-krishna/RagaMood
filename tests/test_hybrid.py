import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hybrid import Index, rrf, search, tokens
from playlist import dupe_key


@pytest.fixture(scope="module")
def ix():
    return Index()


def test_rrf_both_beats_one():
    f = rrf([[1, 2, 3], [1, 3, 2]], [1, 1])
    assert f[1] > f[2] and f[1] > f[3]
    assert rrf([[2, 1], [1, 2]], [3, 1])[2] > rrf([[2, 1], [1, 2]], [3, 1])[1]


def test_sparse_exact_names(ix):
    res = search("Sanjay Subrahmanyan kriti Kalyani", ix, 5, "sparse")["songs"]
    assert res and "kalyani" in tokens(res[0]["raga"] + " " + res[0]["title"])


def test_dense_variant_spelling(ix):
    res = search("sankarabharanam", ix, 10, "dense")["songs"]
    assert any("sankarabharanam" in tokens(s["raga"]) for s in res[:5])


def test_final_list_unique_and_capped(ix):
    res = search("slow classical vocal concert", ix, 15)["songs"]
    assert len(res) == 15
    assert len({dupe_key(type("S", (), {"title": s["title"]})) for s in res}) == 15
    from collections import Counter
    c = Counter(a for s in res for a in s["artists"])
    assert max(c.values()) <= 2


def test_n_honoured_and_modes(ix):
    for m in ("hybrid", "dense", "sparse"):
        assert len(search("hindustani khayal teentaal", ix, 4, m)["songs"]) <= 4
