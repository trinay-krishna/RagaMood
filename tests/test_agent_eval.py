import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import RASAS
from evaluation.cases import load_cases
from evaluation.metrics import build_vocab, entity_ok, keyword_router, pred_tool, score_turns, summarize


def turn(name=None, reply="", **args):
    return {"calls": [{"name": name, "args": args, "result": {}}] if name else [], "reply": reply, "latency": 1.0}


def probs(rasa):
    p = np.full(len(RASAS), 0.01)
    p[RASAS.index(rasa)] = 1.0
    return p / p.sum()


def test_cases_are_wellformed():
    cases = load_cases()
    assert len(cases) >= 100 and len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert c["turns"] and c["gold_tool"]
        assert set(c["gold_rasas"]) <= set(RASAS)


def test_pred_tool():
    assert pred_tool(turn("search_catalog", query="x")) == "search_catalog"
    assert pred_tool(turn(reply="How are you feeling?")) == "clarify"
    assert pred_tool(turn(reply="Hello!")) == "none"


def test_entity_ok_normalises():
    assert entity_ok("Bageshree khayal", ["bageshri", "bageshree"])
    assert not entity_ok("", ["x"]) and not entity_ok("raga Kalyani", ["mohana"])


def test_score_and_summarize():
    cases = [{"id": "a", "category": "inversion", "gold_tool": ["recommend_by_mood"], "gold_rasas": ["Hasya"], "gold_entity": []},
             {"id": "b", "category": "keyword", "gold_tool": ["search_catalog"], "gold_rasas": [], "gold_entity": ["kalyani"]},
             {"id": "c", "category": "chitchat", "gold_tool": ["none"], "gold_rasas": [], "gold_entity": []}]
    turns = [turn("recommend_by_mood", mood_text="I feel happy."), turn("search_catalog", query="Mohana"), turn("recommend_by_mood", mood_text="x")]
    df = score_turns(cases, turns, [probs("Hasya"), None, probs("Karuna")], RASAS)
    assert df["tool_ok"].tolist() == [True, True, False]
    assert df.loc[0, "rewrite_hit1"] and not df.loc[1, "entity_ok"]
    s = summarize(df, "category")
    assert s.loc["ALL", "n"] == 3 and abs(s.loc["ALL", "tool_acc"] - 2 / 3) < 1e-9


def test_keyword_router_has_no_rewriting():
    vocab = build_vocab([], ["Kalyani"])
    t = keyword_router("I'm sad, play raga Kalyani", vocab)
    assert t["calls"][0]["name"] == "recommend_mixed" and t["calls"][0]["args"]["emotion"] == "I'm sad, play raga Kalyani"
    assert keyword_router("hello there", vocab)["calls"] == []
    assert keyword_router("I'm sad, cheer me up", vocab)["calls"][0]["args"]["mood_text"] == "I'm sad, cheer me up"
