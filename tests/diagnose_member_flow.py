"""
tests/diagnose_member_flow.py

Offline diagnostic for the "user says M001, agent stays silent" symptom.

Runs the triage agent end-to-end via ADK's InMemoryRunner (no web UI) and
prints every event — user message, model response, tool call, tool result,
and any error — so you can see exactly where the flow breaks.

Run from MyCode:
    cd C:\\MyStuffs\\MyCode
    python healix/tests/diagnose_member_flow.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Windows consoles default to cp1252 — force UTF-8 so emoji-containing model
# replies can be printed without crashing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logging.getLogger("google_adk").setLevel(logging.DEBUG)
logging.getLogger("google_genai").setLevel(logging.INFO)

from google.adk.runners import InMemoryRunner
from google.genai import types

from healix import root_agent


async def run(runner: InMemoryRunner, user_id: str, session_id: str, text: str) -> None:
    print(f"\n--- USER: {text!r} " + "-" * 40)
    content = types.Content(role="user", parts=[types.Part(text=text)])
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        author = getattr(event, "author", "?")
        if event.content and event.content.parts:
            for p in event.content.parts:
                if getattr(p, "text", None):
                    print(f"  [{author}] text: {p.text[:300]}")
                if getattr(p, "function_call", None):
                    fc = p.function_call
                    print(f"  [{author}] tool_call: {fc.name}({dict(fc.args)})")
                if getattr(p, "function_response", None):
                    fr = p.function_response
                    resp = dict(fr.response) if hasattr(fr.response, "items") else fr.response
                    short = str(resp)[:300]
                    print(f"  [{author}] tool_result({fr.name}): {short}")
        if getattr(event, "error_code", None):
            print(f"  [{author}] ERROR {event.error_code}: {event.error_message}")


async def main() -> None:
    runner = InMemoryRunner(agent=root_agent, app_name="Healix")
    session = await runner.session_service.create_session(
        app_name="Healix", user_id="diag_user"
    )
    sid = session.id

    await run(runner, "diag_user", sid, "Hi")
    await run(runner, "diag_user", sid, "M001")

    # Final state dump
    final = await runner.session_service.get_session(
        app_name="Healix", user_id="diag_user", session_id=sid
    )
    print("\n--- FINAL SESSION STATE " + "-" * 30)
    for k, v in (final.state or {}).items():
        if isinstance(v, dict):
            print(f"  {k}: <dict with keys {list(v.keys())}>")
        else:
            print(f"  {k}: {str(v)[:120]}")
    if not final.state:
        print("  (state is empty)")


if __name__ == "__main__":
    asyncio.run(main())
