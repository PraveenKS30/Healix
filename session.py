"""
healix/session.py

Session persistence layer for Healix.

Uses ADK's DatabaseSessionService with SQLite so that conversation history,
member profiles, and conversation summaries survive process restarts.

Session keying strategy:
  - user_id  = member_id (e.g. "M001")
  - session_id = member_id  (one persistent session per member)

When a member returns, get_or_create_session() finds their existing session and
restores full conversation history + session state automatically.

Usage (programmatic):
    from healix.session import get_session_service, get_or_create_session

    service = get_session_service()
    session = await get_or_create_session(service, "Healix", "M001")

Note for adk web:
    adk web manages its own runner and session service internally.
    The DatabaseSessionService defined here is used for programmatic / production
    runs. For demo purposes, adk web's in-memory sessions are sufficient
    (member profile is re-fetched from get_member_summary at session start).
"""

import os

from google.adk.sessions import DatabaseSessionService

_DB_URL = os.getenv("HEALIX_DB_URL", "sqlite+aiosqlite:///healix.db")


def get_session_service() -> DatabaseSessionService:
    """
    Return a DatabaseSessionService backed by SQLite (or the URL in HEALIX_DB_URL).

    The database file is created automatically on first use.
    """
    return DatabaseSessionService(db_url=_DB_URL)


async def get_or_create_session(
    service: DatabaseSessionService,
    app_name: str,
    member_id: str,
):
    """
    Return an existing session for the member, or create a new one.

    Args:
        service:   DatabaseSessionService instance
        app_name:  ADK app name (e.g. "Healix")
        member_id: Stable member identifier used as both user_id and session_id

    Returns:
        ADK Session object (existing or freshly created)
    """
    session = await service.get_session(
        app_name=app_name,
        user_id=member_id,
        session_id=member_id,
    )
    if session is None:
        session = await service.create_session(
            app_name=app_name,
            user_id=member_id,
            session_id=member_id,
        )
    return session
