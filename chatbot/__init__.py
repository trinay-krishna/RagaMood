"""Conversational front end: an LLM that routes a user's message to playlist / hybrid / mixed.

Only chatbot.tools imports the song pipeline; everything else works with plain dicts.
"""
from .agent import ChatBot

__all__ = ["ChatBot"]
