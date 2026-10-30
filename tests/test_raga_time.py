import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import RASAS
from pipeline import songs_for_text
from playlist import make_playlist
from raga_time import current_slots, parse_time, time_match


def stub(top):
    def f(texts):
        p = np.full(len(RASAS), 0.01)
        p[RASAS.index(top)] = 1.0
        return [p / p.sum()]
    return f


def test_slots_and_wrap():
    assert current_slots(5) == {"Dawn"}
    assert current_slots(6.5) == {"Dawn", "Morning"}
    assert "Night" in current_slots(23) and "Night" in current_slots(0.5)
    assert current_slots(0.5) >= {"Night", "Late night"}
    assert current_slots(7) == {"Morning"}  # end of window is exclusive
    assert parse_time("23:30") == 23.5 and parse_time(None) is None


def test_name_matching():
    assert time_match("Bageshri", 23) and not time_match("Bageshri", 14)
    assert time_match("Madhuvanti", 14)
    assert time_match("YamanKalyani", 17)
    assert time_match("Behaag", 20)
    assert time_match("Lalith", 5)
    assert not time_match("Kalyani", 17)


def test_pipeline_matched_raga_first_rasa_order_kept():
    res = songs_for_text("x", 1, scorer=stub("Karuna"), hour=23)
    ragas = [e["raga"] for e in res["ragas"]]
    assert ragas[0] == "Bageshri" and res["ragas"][0]["time_match"]
    assert sorted(ragas) == sorted(set(ragas))
    res = songs_for_text("x", 1, scorer=stub("Karuna"), hour=14)
    assert res["ragas"][0]["raga"] == "Madhuvanti"


def test_playlist_boosts_matched_raga():
    night = make_playlist("x", n=8, top_k=1, scorer=stub("Karuna"), hour=23)["songs"]
    day = make_playlist("x", n=8, top_k=1, scorer=stub("Karuna"), hour=10)["songs"]
    count = lambda ss: sum(s["raga"] == "Bageshri" for s in ss)
    assert count(night) >= count(day)
    assert night[0]["raga"] == "Bageshri" or count(night) == 0
