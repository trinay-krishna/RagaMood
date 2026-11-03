"""Chat session: history + tool-calling loop. Knows nothing about the song pipeline (see tools.py)."""
import json

from . import config
from .guard import NUDGE, SAFE_REPLY, has_feeling, looks_like_song_list
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
                 max_rounds=config.MAX_TOOL_ROUNDS, guard=True):
        self.llm = llm or OllamaLLM()
        self.tools = tools or Tools()
        self.history_turns, self.max_rounds, self.guard = history_turns, max_rounds, guard
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

    def _needs_nudge(self, text, reply):
        """A reply with no tool call that should have had one: invented songs, a feeling answered in words,
        or a question about a request that already named a raga / artist / form."""
        if looks_like_song_list(reply) or has_feeling(text):
            return True
        hint = getattr(self.tools, "entity_hint", None)
        return bool(hint and hint(text))

    def send(self, text):
        """-> {"reply": str, "tool_calls": [{"name","args","result"}], "songs": [...], "retried": bool, "blocked": bool}"""
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *self._history(),
                {"role": "user", "content": text}]
        calls, songs, reply = [], [], ""
        retried = blocked = False
        for rnd in range(self.max_rounds + 1):
            # after the last round, drop the tools so the model has to answer
            msg = self.llm.chat(msgs, TOOL_SCHEMAS if rnd < self.max_rounds else None)
            tcs = msg.get("tool_calls") or []
            if not tcs:
                reply = (msg.get("content") or "").strip()
                if self.guard and not calls and not retried and rnd < self.max_rounds and self._needs_nudge(text, reply):
                    retried = True                      # transient: the nudge is not kept in the history
                    msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": NUDGE}]
                    continue
                break
            msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
            for tc in tcs:
                fn = tc.get("function", {})
                name, args = fn.get("name", ""), _args(fn.get("arguments"))
                result = {"error": "arguments must be a JSON object"} if args is None else self.tools.call(name, args)
                calls.append({"name": name, "args": args, "result": result})
                songs = result.get("songs", songs)
                msgs.append({"role": "tool", "name": name, "content": json.dumps(result, ensure_ascii=False)})
        if self.guard and not calls and looks_like_song_list(reply):
            reply, blocked = SAFE_REPLY, True           # never show songs the model made up
        reply = reply or "Sorry, I couldn't put a recommendation together. Could you rephrase?"
        self.turns.append({"user": text, "reply": reply,
                           "calls": [f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in calls]})
        return {"reply": reply, "tool_calls": calls, "songs": songs, "retried": retried, "blocked": blocked}
