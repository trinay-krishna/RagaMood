"""Manually try rasa_classifier on your own text.

Interactive:  python rasa_compare/try_classifier.py
One-shot:     python rasa_compare/try_classifier.py "I miss her so much" "What a joke!"
"""
import argparse
import sys

from rasa_classifier import LABELS, predict

sys.stdout.reconfigure(encoding="utf-8")


def show(text, probs, top_k):
    order = probs.argsort()[::-1][:top_k]
    print(" | ".join(f"{LABELS[i]} {probs[i]:.2f}" for i in order))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="*", help="texts to classify; omit for interactive mode")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    if args.texts:
        for text, probs in zip(args.texts, predict(args.texts)):
            print(f"{text}\n  ", end="")
            show(text, probs, args.top_k)
        return

    print("Enter text to classify (blank line or 'quit' to exit).")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text.lower() == "quit":
            break
        show(text, predict([text])[0], args.top_k)


if __name__ == "__main__":
    main()
