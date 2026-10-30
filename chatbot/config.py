"""Chat bot settings; each can be overridden by an environment variable or the CLI."""
import os

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("CHATBOT_MODEL", "qwen3:8b")
TEMPERATURE = 0.3
MAX_TOOL_ROUNDS = 3   # LLM <-> tool iterations per user message
HISTORY_TURNS = 8     # most recent user turns kept in the prompt
DEFAULT_N = 8         # songs per recommendation
TIMEOUT = 300         # seconds per LLM call
