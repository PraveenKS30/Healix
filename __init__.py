"""
healix/

Healix — Multi-Agent AI Health Assistant powered by Google ADK.

This module exposes:
  root_agent      — The HealthAssistantAgent, required by `adk web` for discovery.
  session_service — DatabaseSessionService (SQLite) for persistent conversations.

Usage with adk web:
    cd C:\\MyStuffs\\MyCode
    adk web
    # Select "healix" in the browser UI

Usage programmatic:
    from healix import root_agent, session_service
    from healix.session import get_or_create_session

Environment:
    A .env file placed either in this package directory or in the current working
    directory is auto-loaded at import time (see _load_env below). Existing
    environment variables are NOT overridden, so shell-exported values win.
"""

from pathlib import Path


def _load_env() -> None:
    """
    Load environment variables from a .env file without requiring callers to
    configure anything. Looks in (1) this package's directory, then (2) the
    current working directory. Silent no-op if python-dotenv isn't installed
    or no .env is found.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    candidates = [
        Path(__file__).resolve().parent / ".env",   # healix/.env
        Path.cwd() / ".env",                         # where adk web was launched
    ]
    for path in candidates:
        if path.is_file():
            # override=False → real env vars take precedence over .env values
            load_dotenv(dotenv_path=path, override=False)


_load_env()

from .agents.triage_agent import create_triage_agent  # noqa: E402
from .session import get_session_service  # noqa: E402

# Required by adk web — must be a module-level variable named root_agent
root_agent = create_triage_agent()

# Persistent session service — used for programmatic / production runs
session_service = get_session_service()
