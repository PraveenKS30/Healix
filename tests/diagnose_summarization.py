"""
tests/diagnose_summarization.py

Verify the rolling summarization actually fires past MAX_TURNS and re-fires on
the SUMMARY_INTERVAL cadence. The summarizer is mocked so this runs offline
(no API key, no tokens spent) — what we're verifying is the *trigger logic*,
not the quality of the generated summary.

Outputs after each turn:
  turn N | fired=True/False | stored_summary='...' | last_summarized=N

Run:
    cd C:\\MyStuffs\\MyCode
    python healix/tests/diagnose_summarization.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)

from healix.agents import callbacks as cb  # noqa: E402
from healix.agents import triage_agent as ta  # noqa: E402
from healix.tools.conversation_utils import MAX_TURNS, SUMMARY_INTERVAL  # noqa: E402


def make_event(role: str, text: str):
    return SimpleNamespace(
        content=SimpleNamespace(role=role, parts=[SimpleNamespace(text=text)])
    )


def build_events(n_turns: int) -> list:
    """Alternate user/model turns with plausible clinical content."""
    events = []
    pairs = [
        ("I am Ravi, member M001", "Welcome back, Ravi. How can I help?"),
        ("I've been feeling dizzy in the mornings", "How long has this been happening?"),
        ("About a week now", "Any other symptoms like nausea or headaches?"),
        ("Mild headaches too", "Is your blood pressure being monitored?"),
        ("Yes, 140/90 recently", "That's borderline high — let's talk about it."),
        ("Should I worry?", "It needs attention but is manageable."),
        ("What foods should I avoid?", "Salt, processed foods, and sugary drinks."),
        ("What about coffee?", "Moderate — 1–2 cups daily is fine."),
        ("Metformin timing — before or after meals?", "With meals to reduce GI upset."),
        ("Can I skip breakfast?", "Not recommended with diabetes."),
        ("I walked 30 mins yesterday", "Great — aim for 5 days a week."),
        ("Anything else?", "Track your BP twice a week and review with PCP."),
    ]
    for i in range(n_turns):
        u, m = pairs[i % len(pairs)]
        events.append(make_event("user", u))
        events.append(make_event("model", m))
    return events[:n_turns]  # trim to exact count


def main() -> None:
    state: dict = {}
    summarize_calls: list[int] = []

    def fake_summarize(events: list) -> str:
        n = len(events)
        summarize_calls.append(n)
        return f"[mock summary #{len(summarize_calls)} covering {n} events]"

    print(f"MAX_TURNS={MAX_TURNS}, SUMMARY_INTERVAL={SUMMARY_INTERVAL}")
    print(f"Expected triggers at turn {MAX_TURNS + 1}, {MAX_TURNS + 1 + SUMMARY_INTERVAL}, ...")
    print()
    print(f"{'turn':>4} | {'fired':>5} | {'last':>4} | summary")
    print("-" * 80)

    for turn in range(1, MAX_TURNS + SUMMARY_INTERVAL * 2 + 3):
        events = build_events(turn)
        ctx = SimpleNamespace(
            state=state,
            session=SimpleNamespace(events=events),
            agent_name="HealthAssistantAgent",
        )

        before_calls = len(summarize_calls)
        with patch.object(cb, "summarize_conversation", side_effect=fake_summarize):
            cb.before_agent_callback(ctx)
        fired = len(summarize_calls) > before_calls

        summary = state.get("conversation_summary", "")
        last = state.get("_last_summarized_turn", 0)
        short_summary = (summary[:60] + "...") if len(summary) > 60 else summary
        print(f"{turn:>4} | {str(fired):>5} | {last:>4} | {short_summary}")

    print()
    print(f"Total summarize_conversation calls: {len(summarize_calls)}")
    print(f"Triggered at turn counts: {summarize_calls}")


if __name__ == "__main__":
    main()
