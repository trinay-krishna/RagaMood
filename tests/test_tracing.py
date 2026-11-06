import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chatbot import tracing
from chatbot.agent import ChatBot
from chatbot.tools import Tools
from test_chatbot import FakeLLM, call, say, tools


def test_disabled_by_default_and_wrapper_strips_extra(monkeypatch):
    for k in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert not tracing.enabled()

    @tracing.traceable(name="x")
    def f(a, b=1):
        return a + b

    assert f(1, b=2, langsmith_extra={"metadata": {"k": "v"}}) == 3


def test_needs_both_flag_and_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    assert not tracing.enabled()
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    assert tracing.enabled()


def test_load_env_does_not_override(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text('# c\nA_NEW="1"\nA_OLD=2\n', encoding="utf-8")
    monkeypatch.setenv("A_OLD", "keep")
    monkeypatch.delenv("A_NEW", raising=False)
    tracing.load_env(p)
    import os
    assert os.environ["A_NEW"] == "1" and os.environ["A_OLD"] == "keep"
    monkeypatch.delenv("A_NEW")


def test_chatbot_unchanged_when_tracing_off_and_session_id_resets(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    bot = ChatBot(FakeLLM(call("search_catalog", query="x"), say("done")), tools([]))
    sid = bot.session_id
    out = bot.send("x")
    assert out["reply"] == "done" and out["tool_calls"][0]["name"] == "search_catalog"
    bot.reset()
    assert bot.session_id != sid and isinstance(Tools().call("nope", {}), dict)
