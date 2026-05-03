"""
agents/callbacks.py

Shared ADK callbacks used by the root agent AND all sub-agents.

Keeping them here avoids circular-import issues: sub-agents can't import from
triage_agent (triage_agent imports them), but all agents can safely import from
this module.
"""

import logging

from ..tools.conversation_utils import (
    MAX_TURNS,
    SUMMARY_INTERVAL,
    count_turns,
    summarize_conversation,
)

logger = logging.getLogger("healix.triage")


def before_agent_callback(callback_context) -> None:
    """
    Before-agent callback shared across root and all sub-agents.

    Triggers rolling summarization once the conversation exceeds MAX_TURNS,
    then re-summarizes only after SUMMARY_INTERVAL additional turns accrue.

    The summary is stored in session state under 'conversation_summary' so
    every agent can read it as prior context.  The update is skipped when
    summarize_conversation returns empty, so a transient API failure doesn't
    overwrite a prior good summary.

    Why attach to sub-agents too?
    Once the root agent transfers to a specialist (e.g. SymptomCheckerAgent),
    the root agent's callback never fires again.  Without this the summarizer
    would be blind to any turns handled inside a sub-agent.
    """
    state = callback_context.state
    try:
        session = callback_context.session

        if state.get("_summarizing"):
            return

        events = getattr(session, "events", [])
        turns = count_turns(events)
        last = state.get("_last_summarized_turn", 0)

        if turns > MAX_TURNS and turns - last >= SUMMARY_INTERVAL:
            state["_summarizing"] = True
            logger.info(
                "Summarization firing: turns=%d, last_summarized=%d, interval=%d",
                turns, last, SUMMARY_INTERVAL,
            )
            summary = summarize_conversation(events)
            if summary:
                state["conversation_summary"] = summary
                state["_last_summarized_turn"] = turns
                logger.info(
                    "Summary stored (%d chars): %s",
                    len(summary),
                    summary[:200].replace("\n", " "),
                )
            else:
                logger.warning("Summarization returned empty — keeping prior summary")
    except Exception:
        pass
    finally:
        if state.get("_summarizing"):
            state["_summarizing"] = False
