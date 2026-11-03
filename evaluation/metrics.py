"""Scoring helpers for the agent evaluation (pure functions; no LLM, no model loading here)."""
import re

import numpy as np
import pandas as pd

from chatbot.guard import looks_like_song_list
from raaga_search.search import normalize

TOOLS = ["recommend_by_mood", "search_catalog", "recommend_mixed", "none", "clarify"]


def pred_tool(turn):
    """Tool the agent used in a turn (its last call), else 'clarify' if it asked a question, else 'none'."""
    if turn["calls"]:
        return turn["calls"][-1]["name"]
    return "clarify" if turn["reply"].strip().endswith("?") else "none"


def _arg(turn, *keys):
    for c in reversed(turn["calls"]):
        for k in keys:
            if isinstance(c.get("args"), dict) and c["args"].get(k):
                return str(c["args"][k])
    return ""


def mood_text(turn):
    return _arg(turn, "mood_text", "emotion")


def entity_text(turn):
    return _arg(turn, "query", "entity")


def entity_ok(text, aliases):
    t = normalize(text)
    return bool(t) and any(normalize(a) in t for a in aliases)


def had_error(turn):
    return any("error" in (c.get("result") or {}) for c in turn["calls"])


def score_turns(cases, turns, probs, rasas):
    """cases/turns/probs aligned lists; probs[i] = rasa distribution of mood_text(turns[i]) (or None).
    -> DataFrame with one row per case."""
    rows = []
    for case, turn, p in zip(cases, turns, probs):
        tool = pred_tool(turn)
        row = {"id": case["id"], "category": case["category"], "gold_tool": "|".join(case["gold_tool"]),
               "pred_tool": tool, "tool_ok": tool in case["gold_tool"], "error": had_error(turn),
               "calls": len(turn["calls"]), "latency": turn.get("latency"),
               "fabricated": (not turn["calls"]) and (looks_like_song_list(turn["reply"]) or bool(turn.get("blocked"))),
               "retried": bool(turn.get("retried")), "blocked": bool(turn.get("blocked")), "mood_text": mood_text(turn),
               "entity_text": entity_text(turn)}
        if case["gold_rasas"]:
            order = [rasas[i] for i in np.argsort(-p)] if p is not None else []
            row["rewrite_hit1"] = bool(order and order[0] in case["gold_rasas"])
            row["rewrite_hit3"] = bool(set(order[:3]) & set(case["gold_rasas"]))
        if case["gold_entity"]:
            row["entity_ok"] = entity_ok(row["entity_text"], case["gold_entity"])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df, by=None):
    def agg(g):
        return pd.Series({
            "n": len(g), "tool_acc": g["tool_ok"].mean(),
            "rewrite_hit1": g["rewrite_hit1"].mean() if "rewrite_hit1" in g and g["rewrite_hit1"].notna().any() else np.nan,
            "rewrite_hit3": g["rewrite_hit3"].mean() if "rewrite_hit3" in g and g["rewrite_hit3"].notna().any() else np.nan,
            "entity_acc": g["entity_ok"].mean() if "entity_ok" in g and g["entity_ok"].notna().any() else np.nan,
            "error_rate": g["error"].mean(), "fabricated_rate": g["fabricated"].mean(),
            "retried_rate": g["retried"].mean(), "calls_per_turn": g["calls"].mean(),
            "latency_p50": g["latency"].median(), "latency_p95": g["latency"].quantile(0.95)})
    if by is None:
        return agg(df).to_frame("all").T
    out = df.groupby(by).apply(agg, include_groups=False)
    return pd.concat([out, agg(df).to_frame("ALL").T])


# ----- naive keyword router (no LLM, no query rewriting): the baseline the agent has to beat -----
FORMS = ["kriti", "khayal", "thillana", "varnam", "thumri", "tarana", "bhajan", "mangalam", "viruttham",
         "ragamalika", "teentaal", "javali", "pallavi", "alaap"]
MOOD_RE = re.compile(r"\b(sad|happy|angry|furious|calm|relax\w*|scared|terrified|anxious|love|lonely|miss|joy\w*|"
                     r"brave|courage|amazed|disgust\w*|stress\w*|feel\w*|mood|heartbroken|cheer|upbeat|energ\w*|"
                     r"peace\w*|romantic|worried|down|excited|bored|exhausted)\b", re.I)


def build_vocab(songs, extra_ragas):
    v = {normalize(f) for f in FORMS}
    for s in songs:
        v.update(normalize(a) for a in s.artists)
        v.update(normalize(r) for r in s.raagas)
    v.update(normalize(r) for r in extra_ragas)
    return {x for x in v if len(x) >= 4}


def keyword_router(text, vocab):
    """-> a fake turn dict (same shape the agent logs) built by regex/vocabulary lookup."""
    t = normalize(text)
    ent = any(w in t for w in vocab)
    mood = bool(MOOD_RE.search(text))
    if ent and mood:
        call = {"name": "recommend_mixed", "args": {"emotion": text, "entity": text}}
    elif ent:
        call = {"name": "search_catalog", "args": {"query": text}}
    elif mood:
        call = {"name": "recommend_by_mood", "args": {"mood_text": text}}
    else:
        return {"calls": [], "reply": "", "latency": 0.0}
    return {"calls": [dict(call, result={})], "reply": "", "latency": 0.0}
