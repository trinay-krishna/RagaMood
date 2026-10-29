import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import RASAS
from pipeline import text_to_rasas, songs_for_text
from rasa_raga_map import RASA_TO_RAGAS, search_names


def stub(top):
    def f(texts):
        p = np.full(len(RASAS), 0.01)
        p[RASAS.index(top)] = 1.0
        return [p / p.sum()]
    return f


def test_map_shape():
    assert sum(len(v) for v in RASA_TO_RAGAS.values()) == 35
    assert set(RASA_TO_RAGAS) <= set(RASAS)


def test_unmapped_top_rasa_is_skipped():
    out = text_to_rasas("x", 3, stub("Raudra"))
    assert out and all(r in RASA_TO_RAGAS for r, _ in out)


def test_search_names_split():
    assert {"Abheri", "Bhimpalasi"} <= set(search_names("Abheri/Bhimpalasi"))


def test_songs_found():
    res = songs_for_text("x", 1, scorer=stub("Shringara"))
    assert any(e["raga"] == "Kalyani" and e["songs"] for e in res["ragas"])
