"""python -m chatbot [--model qwen3:8b] [--url http://localhost:11434] [--time HH:MM] [--debug]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from chatbot import config, tracing
from chatbot.agent import ChatBot
from chatbot.llm import LLMError, OllamaLLM
from chatbot.tools import Tools


QUIT_WORDS = {"/quit", "/exit", "quit", "exit", "bye", "goodbye", "q"}


def is_quit(text):
    """True for a bare goodbye / quit command (not for a sentence that merely contains one)."""
    return text.strip().lower().rstrip(".!? ") in QUIT_WORDS


def time_command(text):
    """'/time 23:00' -> ('set', 23.0); '/time now' -> ('clear', None); '/time' -> ('show', None);
    a malformed '/time x' -> ('bad', None); anything else -> None."""
    from raga_time import parse_time
    parts = text.strip().lower().split()
    if not parts or parts[0] != "/time" or len(parts) > 2:
        return None
    if len(parts) == 1:
        return "show", None
    if parts[1] in ("now", "off", "auto", "clear"):
        return "clear", None
    try:
        return "set", parse_time(parts[1])
    except ValueError:
        return "bad", None


def describe_time(hour):
    from raga_time import current_slots
    shown = "the real clock" if hour is None else f"{int(hour):02d}:{int(round(hour % 1 * 60)) % 60:02d}"
    return f"time = {shown}; slots: {', '.join(sorted(current_slots(hour))) or 'none'}"


def main():
    tracing.load_env()
    if tracing.enabled():
        print(f"LangSmith tracing on (project '{tracing.project()}')")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.MODEL)
    ap.add_argument("--url", default=config.OLLAMA_URL)
    ap.add_argument("--time", help="HH:MM for the time-of-day raga boost (default: now)")
    ap.add_argument("--debug", action="store_true", help="show each tool call and its rewritten query")
    a = ap.parse_args()
    hour = None
    if a.time:
        from raga_time import parse_time
        hour = parse_time(a.time)
    bot = ChatBot(OllamaLLM(a.model, a.url), Tools(hour=hour))
    print("Raga music assistant. /reset clears the conversation, /time 23:00 sets the hour for the raga time boost "
          "(/time now = real clock); type bye or /quit to exit.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if is_quit(text):
            break
        if text == "/reset":
            bot.reset()
            print("(conversation cleared)")
            continue
        cmd = time_command(text)
        if cmd:
            kind, h = cmd
            if kind == "bad":
                print("(usage: /time 23:00 sets the hour, /time now uses the real clock, /time shows it)")
                continue
            if kind == "set":
                bot.tools.hour = h
            elif kind == "clear":
                bot.tools.hour = None
            print(f"({describe_time(bot.tools.hour)})")
            continue
        if not text:
            continue
        try:
            out = bot.send(text)
        except LLMError as e:
            print(f"[error] {e}")
            continue
        if a.debug:
            for c in out["tool_calls"]:
                print(f"  [tool] {c['name']}({json.dumps(c['args'], ensure_ascii=False)}) -> "
                      f"{c['result'].get('count', c['result'].get('error'))}")
        print(f"\nbot> {out['reply']}")
    tracing.flush()


if __name__ == "__main__":
    main()
