"""Minimal Ollama chat client. Anything with chat(messages, tools) -> message dict can replace it."""
import requests

from . import config


class LLMError(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, model=None, url=None, temperature=config.TEMPERATURE, timeout=config.TIMEOUT):
        self.model = model or config.MODEL
        self.url = (url or config.OLLAMA_URL).rstrip("/")
        self.temperature, self.timeout = temperature, timeout

    def chat(self, messages, tools=None):
        body = {"model": self.model, "messages": messages, "stream": False, "think": False,
                "options": {"temperature": self.temperature}}
        if tools:
            body["tools"] = tools
        try:
            r = requests.post(f"{self.url}/api/chat", json=body, timeout=self.timeout)
        except requests.ConnectionError as e:
            raise LLMError(f"Cannot reach Ollama at {self.url}. Is it running? (ollama serve)") from e
        if r.status_code == 404:
            raise LLMError(f"Model '{self.model}' not found. Run: ollama pull {self.model}")
        if not r.ok:
            raise LLMError(f"Ollama error {r.status_code}: {r.text[:300]}")
        return r.json()["message"]
