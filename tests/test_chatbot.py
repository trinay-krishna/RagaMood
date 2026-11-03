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
    t = Tools(playlist=mk("recommend_by_mood"), search=mk("search_catalog"), mixed=mk("recommend_mixed"))
    t.entity_hint = lambda text: any(w in text.lower() for w in ("kalyani", "thillana"))   # no catalogue in unit tests
    return t


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


LIST = 'Here you go:\n1. "Kalyani Kriti" - Kalyani - Sanjay Subrahmanyan\n2. "Thillana" - Kalyani - Bombay Jayashri'


def test_guard_retries_when_songs_are_invented_then_calls_tool():
    log = []
    llm = FakeLLM(say(LIST), call("recommend_by_mood", mood_text="I feel joyful."), say("Real songs"))
    bot = ChatBot(llm, tools(log))
    out = bot.send("I can't stop smiling")
    assert out["retried"] and not out["blocked"] and out["reply"] == "Real songs" and log
    assert "must call a tool" in llm.seen[1][0][-1]["content"]          # nudge sent to the model ...
    assert all("must call a tool" not in t["user"] + t["reply"] for t in bot.turns)   # ... but not kept in history


def test_guard_blocks_invented_songs_when_model_keeps_listing():
    out = ChatBot(FakeLLM(say(LIST), say(LIST)), tools([])).send("hello there")
    assert out["blocked"] and "1." not in out["reply"] and not out["tool_calls"]


def test_guard_retries_question_on_keyword_request():
    log = []
    llm = FakeLLM(say("Which raga do you mean?"), call("search_catalog", query="Kalyani"), say("done"))
    out = ChatBot(llm, tools(log)).send("raga Kalyani")
    assert out["retried"] and log == [("search_catalog", {"query": "Kalyani"})]


def test_guard_leaves_chitchat_and_clarifying_questions_alone():
    llm = FakeLLM(say("You're welcome!"), say("How are you feeling?"))
    bot = ChatBot(llm, tools([]))
    assert not bot.send("thanks")["retried"] and not bot.send("play something")["retried"]


def test_guard_can_be_disabled():
    out = ChatBot(FakeLLM(say(LIST)), tools([]), guard=False).send("I can't stop smiling")
    assert not out["retried"] and not out["blocked"] and "Kalyani Kriti" in out["reply"]
