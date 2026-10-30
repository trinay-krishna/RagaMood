import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chatbot.agent import ChatBot
from chatbot.prompts import SYSTEM_PROMPT
from chatbot.tools import TOOL_SCHEMAS, Tools


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def say(text):
    return {"role": "assistant", "content": text}


class FakeLLM:
    """Returns scripted messages in order and records what it was sent."""

    def __init__(self, *script):
        self.script, self.seen = list(script), []

    def chat(self, messages, tools=None):
        self.seen.append((json.loads(json.dumps(messages)), tools))
        return self.script.pop(0)


SONG = {"title": "T", "raga": "Kalyani", "raagas": ["Kalyani"], "artists": ["A"], "concert": "C",
        "form": "Kriti", "tradition": "carnatic", "metadata_path": "/x"}


def tools(log):
    def mk(name):
        def f(**kw):
            log.append((name, kw))
            return {"songs": [SONG], "rasas": [{"rasa": "Karuna", "prob": 0.5}]}
        return f
    return Tools(playlist=mk("recommend_by_mood"), search=mk("search_catalog"), mixed=mk("recommend_mixed"))


def test_routes_and_returns_reply():
    log = []
    bot = ChatBot(FakeLLM(call("recommend_by_mood", mood_text="I feel happy."), say("Here you go")), tools(log))
    out = bot.send("I'm sad, cheer me up")
    assert log == [("recommend_by_mood", {"mood_text": "I feel happy."})]
    assert out["reply"] == "Here you go" and out["songs"][0]["title"] == "T"
    assert "metadata_path" not in out["songs"][0]


def test_mixed_and_catalog_args():
    log = []
    bot = ChatBot(FakeLLM(call("recommend_mixed", emotion="I feel sad.", entity="Ahibhairav"), say("ok"),
                          call("search_catalog", query="thillana"), say("ok")), tools(log))
    bot.send("a")
    bot.send("b")
    assert log == [("recommend_mixed", {"emotion": "I feel sad.", "entity": "Ahibhairav"}),
                   ("search_catalog", {"query": "thillana"})]


def test_tool_result_fed_back():
    llm = FakeLLM(call("search_catalog", query="x"), say("done"))
    ChatBot(llm, tools([])).send("x")
    msgs = llm.seen[1][0]
    assert msgs[-1]["role"] == "tool" and json.loads(msgs[-1]["content"])["count"] == 1


def test_history_carries_tool_calls():
    llm = FakeLLM(call("recommend_by_mood", mood_text="I feel calm."), say("first"), say("second"))
    bot = ChatBot(llm, tools([]))
    bot.send("calm me down")
    bot.send("more like that")
    sent = llm.seen[2][0]
    assert any("recommend_by_mood" in m["content"] and "I feel calm." in m["content"] for m in sent)
    assert sent[-1]["content"] == "more like that"


def test_history_trim_and_reset():
    llm = FakeLLM(*[say("r")] * 4)
    bot = ChatBot(llm, tools([]), history_turns=2)
    for t in "abc":
        bot.send(t)
    # history_turns=2 keeps the 2 most recent stored turns plus the new message
    assert [m["content"] for m in llm.seen[2][0] if m["role"] == "user"] == ["a", "b", "c"]
    bot.send("d")
    assert [m["content"] for m in llm.seen[3][0] if m["role"] == "user"] == ["b", "c", "d"]
    bot.reset()
    assert bot.turns == []


def test_bad_tool_and_args_do_not_crash():
    llm = FakeLLM(call("nope", q=1), call("search_catalog", wrong=1), say("sorry"))
    strict = Tools(search=lambda query, n=8, mode="hybrid": {"songs": []})
    out = ChatBot(llm, strict).send("x")
    errs = [json.loads(m["content"]) for m in llm.seen[-1][0] if m["role"] == "tool"]
    assert all("error" in e for e in errs) and out["reply"] == "sorry"


def test_round_cap_forces_answer():
    llm = FakeLLM(*[call("search_catalog", query="x")] * 3, say("final"))
    out = ChatBot(llm, tools([]), max_rounds=3).send("x")
    assert out["reply"] == "final" and llm.seen[-1][1] is None


def test_prompt_and_schemas():
    for s in ("recommend_by_mood", "search_catalog", "recommend_mixed", "cheer me up"):
        assert s in SYSTEM_PROMPT
    assert {t["function"]["name"] for t in TOOL_SCHEMAS} == {"recommend_by_mood", "search_catalog", "recommend_mixed"}
