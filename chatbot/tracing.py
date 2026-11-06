"""Optional LangSmith tracing for the chat bot.

Off by default. It turns on only when LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set (the CLI loads them from .env),
so tests, the evaluation scripts and library use never send anything. Without the `langsmith` package this is a no-op.
"""
import functools
import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env(path=None):
    """Read KEY=VALUE lines from .env into os.environ (existing variables win)."""
    p = Path(path) if path else ENV_FILE
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def enabled():
    on = (os.environ.get("LANGSMITH_TRACING") or os.environ.get("LANGCHAIN_TRACING_V2") or "").lower() in ("1", "true", "yes")
    return on and bool(os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"))


def project():
    return os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT") or "default"


def traceable(**kw):
    """langsmith.traceable that is decided at call time. The wrapped function also accepts langsmith_extra=... (ignored when off)."""
    def deco(fn):
        real = None

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal real
            extra = kwargs.pop("langsmith_extra", None)
            if not enabled():
                return fn(*args, **kwargs)
            if real is None:
                try:
                    from langsmith import traceable as ls_traceable
                    real = ls_traceable(**kw)(fn)
                except ImportError:
                    real = fn
            if extra and real is not fn:
                kwargs["langsmith_extra"] = extra
            return real(*args, **kwargs)
        return wrapper
    return deco


def without_self(inputs):
    return {k: v for k, v in inputs.items() if k != "self"}


def flush():
    """Send any queued traces (call before the process exits)."""
    if not enabled():
        return
    try:
        from langsmith import Client
        Client().flush()
    except Exception:
        pass
