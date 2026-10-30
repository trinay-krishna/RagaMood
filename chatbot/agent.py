"""Chat session: history + tool-calling loop. Knows nothing about the song pipeline (see tools.py)."""
import json

from . import config
from .llm import OllamaLLM
from .prompts import SYSTEM_PROMPT
from .tools import TOOL_SCHEMAS, Tools


def _args(raw):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    return raw if isinstance(raw, dict) else None


class ChatBot:
    def __init__(self, llm=None, tools=None, history_turns=config.HISTORY_TURNS,
                 max_rounds=config.MAX_TOOL_ROUNDS):
        self.llm = llm or OllamaLLM()
        self.tools = tools or Tools()
        self.history_turns, self.max_rounds = history_turns, max_rounds
        self.turns = []  # [{"user", "calls": [str], "reply"}]

    def reset(self):
        self.turns = []

    def _history(self):
        msgs = []
        for t in self.turns[-self.history_turns:]:
            msgs.append({"role": "user", "content": t["user"]})
            note = "".join(f"[tool call: {c}]\n" for c in t["calls"])
            msgs.append({"role": "assistant", "content": note + t["reply"]})
        return msgs

    def send(self, text):
        """-> {"reply": str, "tool_calls": [{"name","args","result"}], "songs": [...]}"""
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *self._history(),
                {"role": "user", "content": text}]
        calls, songs, reply = [], [], ""
        for rnd in range(self.max_rounds + 1):
            # after the last round, drop the tools so the model has to answer
            msg = self.llm.chat(msgs, TOOL_SCHEMAS if rnd < self.max_rounds else None)
            tcs = msg.get("tool_calls") or []
            if not tcs:
                reply = (msg.get("content") or "").strip()
                break
            msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
            for tc in tcs:
                fn = tc.get("function", {})
                name, args = fn.get("name", ""), _args(fn.get("arguments"))
                result = {"error": "arguments must be a JSON object"} if args is None else self.tools.call(name, args)
                calls.append({"name": name, "args": args, "result": result})
                songs = result.get("songs", songs)
                msgs.append({"role": "tool", "name": name, "content": json.dumps(result, ensure_ascii=False)})
        reply = reply or "Sorry, I couldn't put a recommendation together. Could you rephrase?"
        self.turns.append({"user": text, "reply": reply,
                           "calls": [f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in calls]})
        return {"reply": reply, "tool_calls": calls, "songs": songs}
