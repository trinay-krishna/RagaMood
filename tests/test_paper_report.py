import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))

import pandas as pd

from make_paper_report import best_mask, build_tables, keyword_tables, mood_tables, render_html


def fixture(tmp_path):
    rr, ev = tmp_path / "rerank", tmp_path / "eval"
    rr.mkdir()
    ev.mkdir()
    cols = {"top1": [.5, .8], "top3": [.8, .9], "macro_f1": [.5, .8], "top1_merged": [.5, .8]}
    pd.DataFrame(cols, index=["clf", "fusion"]).to_csv(rr / "cv_metrics.csv")
    pd.DataFrame(cols, index=["clf", "vad+clf"]).to_csv(rr / "ablation.csv")
    pd.DataFrame({"system": ["agent_v2", "oracle"], "tool_acc": [.9, 1.], "mood_precision": [.5, .7], "mood_ndcg": [.5, .6],
                  "entity_hit": [.9, 1.]}).set_index("system").to_csv(ev / "e2e_comparison.csv")
    return rr, ev


def test_tables_built_and_missing_sources_skipped(tmp_path):
    rr, ev = fixture(tmp_path)
    tables = build_tables(rr, ev)
    assert [t["id"] for t in tables] == ["T1", "T2", "T7"]          # no search / agent files -> skipped, no crash
    assert "oracle (perfect call)" in tables[-1]["df"].index
    assert "e2e_success" not in tables[-1]["df"].columns             # the composite metric is not reported
    html = render_html(tables)
    assert "T7" in html and "Limitations" not in html


def test_keyword_table_is_compact_and_mood_tables_are_split():
    hard = {"target": [{"query": q, "mode": m, "n": 10, "recall@1": .5, "recall@5": .6, "recall@10": .7, "mrr": .6, "ndcg@10": .6 + .1 * i}
                       for q in ("partial title", "two typos", "spelling variant", "artist + raga", "form + raga + artist")
                       for i, m in enumerate(("sparse", "dense", "hybrid"))],
            "set": [{"query": q, "mode": m, "n": 10, "precision@10": .2, "ndcg@10": .5, "top1_relevant": .4}
                    for q in ("raga only", "artist + form", "tradition + form + raga") for m in ("sparse", "dense", "hybrid")]}
    easy = {"known_item": [{"query": q, "mode": m, "n": 10, "recall@1": 1., "recall@5": 1., "recall@10": 1., "mrr": 1., "ndcg@10": 1.}
                           for q in ("title", "title+artist", "title (typo)") for m in ("sparse", "dense", "hybrid")]}
    t3, a3a, a3b = keyword_tables(hard, easy)
    assert len(t3["df"]) <= 10 and sum(c.startswith("nDCG@10") for c in t3["df"].columns) == 3
    assert "title+artist" not in " ".join(t3["df"].index) and a3a["appendix"] and a3b["appendix"]
    mood = {"mood_playlists": {"n_queries": 12, "precision@10": {"playlist": .7, "dense": .03, "random": .03},
                               "ndcg@10": {"playlist": .8, "dense": .04, "random": .04}, "empty_rate": 0.,
                               "per_rasa": [{"rasa": "Karuna", "playlist": .7, "dense": .0, "random": .1, "ndcg_playlist": .8,
                                             "ndcg_dense": .0, "ndcg_random": .1}],
                               "diversity": {"artists": 20., "ragas": 7., "forms": 4., "dupes": 0., "size": 10.}},
            "mixed": {"n_queries": 5, "requested_share": .3, "requested_entity_hit": 1., "mood_section_precision@rasa": .7},
            "time_boost": [{"hour": 5, "boost": .1, "no boost": .05}]}
    ids = [t["id"] for t in mood_tables(mood)]
    assert ids == ["T4", "T4c", "T5"]
    mask = best_mask(t3["df"], t3["shade"])
    assert mask.to_numpy().any()
