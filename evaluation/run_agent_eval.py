"""Run the chat agent on evaluation/cases.py and log every turn (resumable).

Run:  python evaluation/run_agent_eval.py [--model qwen3:8b] [--limit N] [--fresh] [--out evaluation/results/agent_log.jsonl]
Scoring and plots happen in report.py from the log, so the slow LLM run is done once.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from chatbot import config
from chatbot.agent import ChatBot
from chatbot.llm import OllamaLLM
from chatbot.tools import Tools
from evaluation.cases import load_cases


class RoutingOnlyLLM:
    """Wraps the real LLM but answers the post-tool call with a stub: routing and query rewriting are decided in the
    first call, and writing the reply is slow on CPU. Use --full-reply for the real thing."""

    def __init__(self, llm):
        self.llm = llm

    def chat(self, messages, tools=None):
        if messages[-1]["role"] == "tool":
            return {"role": "assistant", "content": "Here are some songs."}
        return self.llm.chat(messages, tools)


def slim(result):
    return {k: result[k] for k in ("count", "error", "note", "rasas") if k in result}


def run_case(case, llm, tools, guard=True):
    bot = ChatBot(llm, tools, guard=guard)
    turns = []
    for text in case["turns"]:
        t0 = time.time()
        out = bot.send(text)
        turns.append({"user": text, "reply": out["reply"], "latency": round(time.time() - t0, 2),
                      "retried": out["retried"], "blocked": out["blocked"],
                      "calls": [{"name": c["name"], "args": c["args"], "result": slim(c["result"])}
                                for c in out["tool_calls"]]})
    return {"id": case["id"], "turns": turns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.MODEL)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--full-reply", action="store_true", help="also generate the final reply after tool results (slow)")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing log")
    ap.add_argument("--tag", default="v2", help="log name: results/agent_log_<tag>.jsonl")
    ap.add_argument("--no-guard", action="store_true", help="prompt only: disable the retry/safe-reply guard")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = Path(a.out or HERE / "results" / f"agent_log_{a.tag}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and not a.fresh:
        done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    elif out.exists():
        out.unlink()
    cases = [c for c in load_cases() if c["id"] not in done][:a.limit]
    llm, tools = OllamaLLM(a.model), Tools()
    if not a.full_reply:
        llm = RoutingOnlyLLM(llm)
    print(f"{len(done)} done, running {len(cases)} cases with {a.model}")
    with out.open("a", encoding="utf-8") as f:
        for i, case in enumerate(cases, 1):
            rec = run_case(case, llm, tools, guard=not a.no_guard)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            last = rec["turns"][-1]
            used = last["calls"][-1]["name"] if last["calls"] else "-"
            print(f"[{i}/{len(cases)}] {case['id']}: {used}  {last['latency']}s  | {case['turns'][-1][:60]}")


if __name__ == "__main__":
    main()
