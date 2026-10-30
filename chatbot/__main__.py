"""python -m chatbot [--model qwen3:8b] [--url http://localhost:11434] [--time HH:MM] [--debug]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from chatbot import config
from chatbot.agent import ChatBot
from chatbot.llm import LLMError, OllamaLLM
from chatbot.tools import Tools


def main():
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
    print("Raga music assistant. /reset clears the conversation, /quit exits.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in ("/quit", "/exit"):
            break
        if text == "/reset":
            bot.reset()
            print("(conversation cleared)")
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


if __name__ == "__main__":
    main()
