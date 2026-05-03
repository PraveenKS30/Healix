"""
tools/conversation_utils.py

Utilities for managing long conversation context in the Healix agents.

Strategy: When a session exceeds MAX_TURNS, the before_agent_callback on the
root agent calls summarize_conversation() to compress the history into a short
summary stored in session state. All agents read this summary for prior context,
keeping each LLM call small regardless of conversation length.
"""

import os

from google import genai

MAX_TURNS = 5         # Summarize after this many user/assistant turns
SUMMARY_INTERVAL = 10  # Re-summarize only every N new turns past MAX_TURNS
SUMMARY_MODEL = "gemini-2.5-flash"
_TRANSCRIPT_WINDOW = 40  # How many recent lines to feed the summarizer
_LINE_CHAR_CAP = 400     # Per-line truncation before sending


def count_turns(events: list) -> int:
    """Count user + model turns in an ADK event list."""
    return sum(
        1 for e in events
        if hasattr(e, "content") and e.content and e.content.role in ("user", "model")
    )


def _build_transcript(events: list) -> tuple[list[str], str]:
    """Return (all_lines, windowed_transcript) from ADK events."""
    lines: list[str] = []
    for event in events:
        if not (hasattr(event, "content") and event.content and event.content.parts):
            continue
        role = getattr(event.content, "role", "")
        if role not in ("user", "model"):
            continue
        text = " ".join(
            p.text for p in event.content.parts if hasattr(p, "text") and p.text
        ).strip()
        if text:
            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {text[:_LINE_CHAR_CAP]}")
    transcript = "\n".join(lines[-_TRANSCRIPT_WINDOW:])
    return lines, transcript


def _make_client() -> genai.Client:
    """
    Build a google.genai Client using either Vertex AI or AI Studio,
    whichever the environment is configured for.

    - Vertex AI:  set GOOGLE_GENAI_USE_VERTEXAI=true plus GOOGLE_CLOUD_PROJECT /
                  GOOGLE_CLOUD_LOCATION (ADC handles auth).
    - AI Studio:  set GOOGLE_API_KEY.
    """
    if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes"):
        return genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set (or GOOGLE_GENAI_USE_VERTEXAI).")
    return genai.Client(api_key=api_key)


def summarize_conversation(events: list) -> str:
    """
    Summarize a list of ADK conversation events into a compact clinical context string.

    Uses google.genai directly (not an ADK agent) to avoid session recursion.
    Respects the same GOOGLE_API_KEY / Vertex AI env the ADK runtime uses,
    so no separate SDK or credentials are needed.

    Returns an empty string on failure so the caller can decide not to
    overwrite an existing (good) summary with a useless placeholder.
    """
    lines, transcript = _build_transcript(events)
    if not lines:
        return ""

    prompt = (
        "You are a clinical documentation assistant. "
        "Summarize the following health conversation in 150 words or less. "
        "Focus on: the user's health concerns raised, key information shared "
        "(age, conditions, medications, symptoms), and any advice or conclusions given. "
        "Write in third person (e.g. 'The user reported...').\n\n"
        f"Conversation:\n{transcript}\n\nSummary:"
    )

    try:
        client = _make_client()
        response = client.models.generate_content(
            model=SUMMARY_MODEL,
            contents=prompt,
        )
        text = (getattr(response, "text", "") or "").strip()
        return text
    except Exception:
        return ""
