"""End-to-end recommendation accuracy: user message -> (agent's tool call) -> songs, scored against the case gold.

Run:  python evaluation/run_e2e_eval.py [--tags v1 v2]     (no LLM: the logged tool calls are replayed through the real,
deterministic pipelines, so the 109-case LLM run is not repeated)
-> evaluation/results/e2e_metrics.json, e2e_comparison.csv, e2e_scored.csv

Systems on identical cases:
  agent_<tag>   the logged tool name + rewritten arguments of the Qwen3 agent
  keyword       the regex router (metrics.keyword_router), no rewriting
  oracle        gold tool + canonical gold-rasa sentence / gold entity: the best the retrieval side can do
Gap between agent and oracle = what the LLM's routing and rewriting cost.

Relevance: a mood song is relevant if its raga is in the paper's map for any gold rasa (the pipeline retrieves with the
same map, so mood numbers measure consistency with it, not listener agreement). An entity song is relevant if its
title / raga / artists / form / concert contains a gold alias. Cases without a scorable part (no tool expected, or only
rasas with no ragas) count as correct when the tool choice is correct.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from chatbot.tools import Tools
from evaluation.cases import CAT, MIX, MOOD, load_cases
from evaluation.metrics import build_vocab, keyword_router, pred_tool
from raaga_search.search import normalize
from rasa_raga_map import RASA_TO_RAGAS
from run_retrieval_eval import ndcg, rasa_names

RES = HERE / "results"
SENTENCE = {"Karuna": "I feel sad and sorrowful.", "Hasya": "I feel happy and playful.",
            "Shringara": "I feel romantic and in love.", "Shanta": "I feel calm and peaceful.",
            "Veera": "I feel brave and determined.", "Adbhuta": "I feel amazed and full of wonder."}
PASS_PRECISION = 0.5          # a mood playlist "works" when at least this share of its songs fits the gold rasa


def load_log(tag):
    path = RES / f"agent_log_{tag}.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines() if path.exists() else [])}


def mappable(case):
    return [r for r in case["gold_rasas"] if r in RASA_TO_RAGAS]


def raga_list(song):
    return [x.strip() for x in str(song.get("raga") or "").split(",") if x.strip()]


def mood_rel(song, rasas):
    names = set().union(*(rasa_names(r) for r in rasas))
    return int(any(normalize(r) in names for r in raga_list(song)))


def entity_rel(song, aliases):
    blob = normalize(" ".join([str(song.get("title") or ""), str(song.get("raga") or ""), " ".join(song.get("artists") or []),
                               str(song.get("form") or ""), str(song.get("concert") or "")]))
    return int(any(normalize(a) in blob for a in aliases))


def oracle_turn(case):
    """The call a perfect agent would make for this case (None for tool-less cases)."""
    tool = case["gold_tool"][0]
    ras = mappable(case) or ["Karuna"]
    sent = SENTENCE.get(ras[0], "I feel calm.")
    ent = case["gold_entity"][0] if case["gold_entity"] else ""
    if tool == MOOD:
        args = {"mood_text": sent}
    elif tool == CAT:
        args = {"query": ent}
    elif tool == MIX:
        args = {"emotion": sent, "entity": ent}
    else:
        return {"calls": [], "reply": "Could you tell me more?" if tool == "clarify" else "ok", "latency": 0.0}
    return {"calls": [{"name": tool, "args": args}], "reply": "", "latency": 0.0}


def replay(tools, turn):
    """-> songs returned by the turn's last tool call ([] if none / error)."""
    songs = []
    for c in turn["calls"]:
        res = tools.call(c["name"], c["args"] or {})
        songs = res.get("songs", songs)
    return songs


def score(case, turn, songs, n_rel_mood):
    tool = pred_tool(turn)
    row = {"id": case["id"], "category": case["category"], "gold_tool": "|".join(case["gold_tool"]),
           "pred_tool": tool, "tool_ok": tool in case["gold_tool"], "n_songs": len(songs)}
    ras = mappable(case)
    row["mood_scored"] = bool(ras) and any(t in case["gold_tool"] for t in (MOOD, MIX))
    row["entity_scored"] = bool(case["gold_entity"]) and any(t in case["gold_tool"] for t in (CAT, MIX))
    ok = row["tool_ok"]
    if row["mood_scored"]:
        # in a mixed answer the 'requested' section is the keyword part; the mood part is judged on the rest
        rel = [mood_rel(s, ras) for s in songs if s.get("section") != "requested"]
        row["mood_precision"] = float(np.mean(rel)) if rel else 0.0
        row["mood_ndcg"] = ndcg(rel, n_rel_mood[tuple(ras)])
        ok = ok and row["mood_precision"] >= PASS_PRECISION
    if row["entity_scored"]:
        rel = [entity_rel(s, case["gold_entity"]) for s in songs]
        row["entity_precision"] = float(np.mean(rel)) if rel else 0.0
        row["entity_hit"] = float(any(rel))
        ok = ok and bool(any(rel))
    row["e2e_ok"] = bool(ok)
    return row


def summarize(df):
    def agg(g):
        return pd.Series({
            "n": len(g), "tool_acc": g["tool_ok"].mean(), "e2e_success": g["e2e_ok"].mean(),
            "mood_n": int(g["mood_scored"].sum()),
            "mood_precision": g.loc[g["mood_scored"], "mood_precision"].mean(),
            "mood_ndcg": g.loc[g["mood_scored"], "mood_ndcg"].mean(),
            "entity_n": int(g["entity_scored"].sum()),
            "entity_hit": g.loc[g["entity_scored"], "entity_hit"].mean(),
            "entity_precision": g.loc[g["entity_scored"], "entity_precision"].mean()})
    return df.groupby("system").apply(agg, include_groups=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["v1", "v2"])
    args = ap.parse_args()
    cases = load_cases()
    tools = Tools()
    ix = tools._ix()
    songs_all = [s.to_dict() for s in ix.songs]
    vocab = build_vocab(ix.songs, [r for rs in RASA_TO_RAGAS.values() for r in rs])
    n_rel_mood = {}
    for c in cases:
        ras = tuple(mappable(c))
        if ras and ras not in n_rel_mood:
            n_rel_mood[ras] = sum(mood_rel({"raga": ", ".join(s.get("raagas") or [])}, ras) for s in songs_all)

    systems = {f"agent_{t}": (lambda c, log=load_log(t): (log.get(c["id"]) or {"turns": []})["turns"][-1:] or None)
               for t in args.tags}
    rows = []
    for name, getter in {**systems, "keyword": None, "oracle": None}.items():
        for c in cases:
            if name == "keyword":
                turn = keyword_router(c["turns"][-1], vocab)
            elif name == "oracle":
                turn = oracle_turn(c)
            else:
                got = getter(c)
                if not got:
                    continue
                turn = got[0]
            songs = replay(tools, turn)
            rows.append({"system": name, **score(c, turn, songs, n_rel_mood)})
        print("done", name, flush=True)
    df = pd.DataFrame(rows)
    comp = summarize(df)
    by_cat = df.groupby(["system", "category"])[["tool_ok", "e2e_ok"]].mean().round(3)
    df.to_csv(RES / "e2e_scored.csv", index=False)
    comp.round(3).to_csv(RES / "e2e_comparison.csv")
    (RES / "e2e_metrics.json").write_text(json.dumps(
        {"overall": comp.round(4).reset_index().to_dict("records"),
         "by_category": by_cat.reset_index().to_dict("records")}, indent=2), encoding="utf-8")
    print(comp.round(3).to_string())
    print(by_cat.unstack(0).round(2).to_string())


if __name__ == "__main__":
    main()
