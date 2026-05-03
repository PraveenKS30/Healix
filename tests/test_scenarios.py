"""
tests/test_scenarios.py

Self-contained regression tests for Healix's pure-logic surfaces.
No pytest dependency — runs with plain `python`.

Coverage:
  1. Safety guardrail — red-flag regex, callback short-circuit, priority ordering
  2. Drug-name canonicalization — offline fallback + RxNorm happy/sad path (mocked)
  3. Drug-interaction early exits
  4. Member-summary lookup — exact ID, partial name, unknown
  5. Conversation utilities — count_turns, transcript building, empty-summary
  6. Triage before_callback — threshold, interval gating, empty-summary protection
  7. after_tool_callback — member_profile persistence, tool-name filter
  8. Root-agent wiring — sub-agents and safety_guard attached everywhere

Network calls (RxNorm, OpenFDA, MedlinePlus, PubMed, SNOMED) are MOCKED so the
suite is deterministic and offline-safe.

Run from anywhere:
    python healix/tests/test_scenarios.py      # from MyCode/
    python tests/test_scenarios.py                    # from healix/
    python -m healix.tests.test_scenarios      # from MyCode/
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Make the package importable regardless of where we're invoked from.
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]  # ...\MyCode
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from healix import root_agent  # noqa: E402
from healix.agents import callbacks as cb  # noqa: E402
from healix.agents import triage_agent as ta  # noqa: E402
from healix.tools.conversation_utils import (  # noqa: E402
    MAX_TURNS,
    SUMMARY_INTERVAL,
    _build_transcript,
    count_turns,
    summarize_conversation,
)
from healix.tools.health_apis import (  # noqa: E402
    _canonical_drug_name,
    _drug_base_fallback,
    fetch_drug_interactions,
    get_member_summary,
)
from healix.tools.safety import (  # noqa: E402
    safety_guard,
    scan_for_red_flags,
)


# ─── Tiny harness ────────────────────────────────────────────────────────────

_TESTS: list = []


def test(fn):
    _TESTS.append(fn)
    return fn


# ─── Fixtures / helpers ──────────────────────────────────────────────────────

def _make_request(text: str):
    part = SimpleNamespace(text=text)
    content = SimpleNamespace(role="user", parts=[part])
    return SimpleNamespace(contents=[content])


def _make_event(role: str, text: str):
    return SimpleNamespace(
        content=SimpleNamespace(role=role, parts=[SimpleNamespace(text=text)])
    )


def _make_ctx(state: dict | None = None, events: list | None = None):
    return SimpleNamespace(
        state=state if state is not None else {},
        session=SimpleNamespace(events=events or []),
        agent_name="TestAgent",
    )


def _mock_rxnorm_response(canonical_name: str, tty: str = "IN"):
    """Return a MagicMock httpx.Client that answers RxNorm with `canonical_name`."""
    client = MagicMock()
    resp = MagicMock()
    resp.json.return_value = {
        "drugGroup": {
            "conceptGroup": [
                {"tty": tty, "conceptProperties": [{"name": canonical_name}]},
            ]
        }
    }
    resp.raise_for_status = MagicMock()
    client.get.return_value = resp
    return client


# ══════════════════════════════════════════════════════════════════════════════
# 1. Safety guardrail
# ══════════════════════════════════════════════════════════════════════════════

@test
def safety_each_category_matches_trigger():
    cases = [
        ("crushing chest pain radiating to left arm", "cardiac"),
        ("my face is drooping and speech is slurred", "stroke"),
        ("I cant breathe and lips are turning blue", "respiratory"),
        ("throat is closing", "anaphylaxis"),
        ("uncontrolled bleeding that wont stop", "severe_bleeding"),
        ("I want to kill myself", "suicidal_ideation"),
        ("I think I overdosed on my pills", "overdose_poisoning"),
    ]
    for text, cat in cases:
        flag = scan_for_red_flags(text)
        assert flag is not None, f"{text!r} should match a category"
        assert flag.category == cat, f"{text!r} -> {flag.category} (expected {cat})"


@test
def safety_expanded_phrasings_match():
    cases = [
        ("i wanna die", "suicidal_ideation"),
        ("im just gonna unalive myself", "suicidal_ideation"),
        ("kms", "suicidal_ideation"),
        ("i have been cutting myself", "suicidal_ideation"),
        ("ive been hurting myself", "suicidal_ideation"),
        ("i engage in self-harm", "suicidal_ideation"),
        ("there is pressure in my chest", "cardiac"),
        ("tightness in the chest for an hour", "cardiac"),
        ("feels like an elephant sitting on my chest", "cardiac"),
        ("my arm went numb suddenly", "stroke"),
        ("my face feels numb", "stroke"),
        ("i cant catch my breath", "respiratory"),
        ("struggling to breathe", "respiratory"),
    ]
    for text, cat in cases:
        flag = scan_for_red_flags(text)
        assert flag is not None, f"{text!r} should match"
        assert flag.category == cat, f"{text!r} -> {flag.category} (expected {cat})"


@test
def safety_no_false_positives_on_benign_input():
    benign = [
        "whats a good diet for diabetes",
        "I have a mild headache",
        "I am Ravi",
        "can I take ibuprofen with lisinopril",
        "Can I cut my fingernails while on blood thinners?",
        "my grandfather died last year",         # 'die' without 'want to die'
        "hypertension runs in my family",
        "",
    ]
    for text in benign:
        flag = scan_for_red_flags(text)
        assert flag is None, f"false positive on {text!r}: {flag}"


@test
def safety_priority_suicidal_beats_overdose():
    # Suicidal-ideation is listed first in _RED_FLAGS, so it must win when both match.
    text = "I took too many pills because I want to kill myself"
    flag = scan_for_red_flags(text)
    assert flag.category == "suicidal_ideation"


@test
def safety_guard_short_circuits_and_records_state():
    ctx = _make_ctx()
    resp = safety_guard(ctx, _make_request("I am having chest pain and my left arm hurts"))
    assert resp is not None, "cardiac should short-circuit"
    body = resp.content.parts[0].text
    assert "911" in body and "heart attack" in body.lower()
    assert ctx.state.get("_last_red_flag") == "cardiac"


@test
def safety_guard_falls_through_on_benign():
    ctx = _make_ctx()
    resp = safety_guard(ctx, _make_request("what foods are good for hypertension"))
    assert resp is None
    assert "_last_red_flag" not in ctx.state


@test
def safety_guard_handles_empty_and_malformed():
    # Empty contents
    assert safety_guard(_make_ctx(), SimpleNamespace(contents=[])) is None
    # Missing contents attribute entirely
    assert safety_guard(_make_ctx(), SimpleNamespace()) is None
    # Non-user-role content only
    req = SimpleNamespace(contents=[
        SimpleNamespace(role="model", parts=[SimpleNamespace(text="chest pain")])
    ])
    # Should NOT scan the model role (it's not the user)
    assert safety_guard(_make_ctx(), req) is None


@test
def safety_guard_picks_most_recent_user_message():
    # Earlier benign user message, later red-flag user message — the latter wins.
    req = SimpleNamespace(contents=[
        SimpleNamespace(role="user", parts=[SimpleNamespace(text="hi")]),
        SimpleNamespace(role="model", parts=[SimpleNamespace(text="hello there")]),
        SimpleNamespace(role="user", parts=[SimpleNamespace(text="I am having a stroke")]),
    ])
    resp = safety_guard(_make_ctx(), req)
    assert resp is not None
    assert "stroke" in resp.content.parts[0].text.lower()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Drug-name canonicalization
# ══════════════════════════════════════════════════════════════════════════════

@test
def drug_base_fallback_strips_dosage_and_prefixes():
    cases = [
        ("Metformin 500mg twice daily", "metformin"),
        ("Sustained-release metformin 500mg", "metformin"),
        ("Extended-release amlodipine", "amlodipine"),
        ("500mg amoxicillin", "amoxicillin"),
        ("Aspirin 81mg once daily", "aspirin"),
        ("Levothyroxine 50mcg", "levothyroxine"),
        ("ER metoprolol", "metoprolol"),
        ("", ""),
    ]
    for raw, expected in cases:
        got = _drug_base_fallback(raw)
        assert got == expected, f"{raw!r} -> {got!r} (expected {expected!r})"


@test
def canonical_drug_name_via_rxnorm_ingredient():
    client = _mock_rxnorm_response("metformin", tty="IN")
    assert _canonical_drug_name(client, "Metformin 500mg") == "metformin"


@test
def canonical_drug_name_prefers_ingredient_over_brand():
    # Response has both BN and IN — IN must win (earlier in _RXNORM_TTY_PREF).
    client = MagicMock()
    resp = MagicMock()
    resp.json.return_value = {
        "drugGroup": {
            "conceptGroup": [
                {"tty": "BN", "conceptProperties": [{"name": "Glucophage"}]},
                {"tty": "IN", "conceptProperties": [{"name": "metformin"}]},
            ]
        }
    }
    resp.raise_for_status = MagicMock()
    client.get.return_value = resp
    assert _canonical_drug_name(client, "Glucophage") == "metformin"


@test
def canonical_drug_name_falls_back_when_rxnorm_fails():
    client = MagicMock()
    client.get.side_effect = RuntimeError("network down")
    # Falls back to offline token parser
    assert _canonical_drug_name(client, "Metformin 500mg") == "metformin"


@test
def canonical_drug_name_falls_back_on_empty_response():
    client = MagicMock()
    resp = MagicMock()
    resp.json.return_value = {"drugGroup": {"conceptGroup": []}}
    resp.raise_for_status = MagicMock()
    client.get.return_value = resp
    assert _canonical_drug_name(client, "Amlodipine 5mg") == "amlodipine"


@test
def canonical_drug_name_empty_input():
    assert _canonical_drug_name(MagicMock(), "") == ""


# ══════════════════════════════════════════════════════════════════════════════
# 3. Drug-interaction early exits
# ══════════════════════════════════════════════════════════════════════════════

@test
def interactions_needs_two_drugs():
    r = fetch_drug_interactions([])
    assert r["interactions"] == []
    assert "at least 2" in r["message"]
    assert "disclaimer" in r

    r = fetch_drug_interactions(["Metformin"])
    assert "at least 2" in r["message"]


@test
def interactions_unparseable_names_return_message():
    # "123mg" and "500mg" have no drug token; fallback returns ""
    # Patch both the client-based canonicalizer and network so we stay offline.
    with patch("healix.tools.health_apis._canonical_drug_name", return_value=""):
        r = fetch_drug_interactions(["123mg", "500mg"])
    assert r["interactions"] == []
    assert "Could not parse" in r["message"]


# ══════════════════════════════════════════════════════════════════════════════
# 4. Member-summary lookup
# ══════════════════════════════════════════════════════════════════════════════

@test
def member_exact_id_lookup():
    r = get_member_summary("M001")
    assert r["name"] == "Ravi Shankar"
    assert "Type 2 Diabetes Mellitus" in r["known_conditions"]


@test
def member_id_is_case_insensitive():
    r = get_member_summary("m001")
    assert r["member_id"] == "M001"


@test
def member_partial_name_match():
    r = get_member_summary("Ravi")
    assert r["member_id"] == "M001"
    r = get_member_summary("chen")  # lowercase partial
    assert r["member_id"] == "M002"


@test
def member_unknown_returns_guest():
    r = get_member_summary("asdfghjk")
    assert r["member_id"] == "GUEST"
    assert r["known_conditions"] == []
    assert r["current_medications"] == []


# ══════════════════════════════════════════════════════════════════════════════
# 5. Conversation utilities
# ══════════════════════════════════════════════════════════════════════════════

@test
def count_turns_basic():
    events = [
        _make_event("user", "hi"),
        _make_event("model", "hello"),
        _make_event("user", "help me"),
    ]
    assert count_turns(events) == 3
    assert count_turns([]) == 0


@test
def build_transcript_filters_non_dialog_events():
    events = [
        _make_event("user", "hi"),
        _make_event("tool", "fake tool payload"),  # should be filtered
        _make_event("model", "hello"),
        SimpleNamespace(content=None),  # malformed — should be skipped
    ]
    lines, transcript = _build_transcript(events)
    assert len(lines) == 2
    assert lines[0].startswith("User: ")
    assert lines[1].startswith("Assistant: ")
    assert "fake tool payload" not in transcript


@test
def summarize_empty_events_returns_empty():
    assert summarize_conversation([]) == ""


# ══════════════════════════════════════════════════════════════════════════════
# 6. Triage before_callback — threshold + interval + empty-summary protection
# ══════════════════════════════════════════════════════════════════════════════

def _events(n: int) -> list:
    return [_make_event("user", f"u{i}") for i in range(n)]


# First fire needs BOTH conditions: turns > MAX_TURNS AND turns >= SUMMARY_INTERVAL
_FIRST_FIRE = max(MAX_TURNS + 1, SUMMARY_INTERVAL)


@test
def callback_below_threshold_no_summarize():
    ctx = _make_ctx(events=_events(max(MAX_TURNS - 5, 0)))
    with patch.object(cb, "summarize_conversation") as mock_sum:
        cb.before_agent_callback(ctx)
    assert mock_sum.call_count == 0
    assert "conversation_summary" not in ctx.state


@test
def callback_crosses_threshold_fires_once():
    ctx = _make_ctx(events=_events(_FIRST_FIRE))
    with patch.object(cb, "summarize_conversation", return_value="SUMMARY-A"):
        cb.before_agent_callback(ctx)
    assert ctx.state["conversation_summary"] == "SUMMARY-A"
    assert ctx.state["_last_summarized_turn"] == _FIRST_FIRE


@test
def callback_interval_gates_repeat_summarization():
    # First fire at _FIRST_FIRE turns
    ctx = _make_ctx(events=_events(_FIRST_FIRE))
    with patch.object(cb, "summarize_conversation", return_value="S1"):
        cb.before_agent_callback(ctx)

    # Within interval: no summarization
    calls = 0

    def counting_summary(_events):
        nonlocal calls
        calls += 1
        return f"S{calls}"

    for turn in range(_FIRST_FIRE + 1, _FIRST_FIRE + SUMMARY_INTERVAL):
        ctx.session.events = _events(turn)
        with patch.object(cb, "summarize_conversation", side_effect=counting_summary):
            cb.before_agent_callback(ctx)
    assert calls == 0, f"expected 0 summarize calls within interval, got {calls}"

    # At _FIRST_FIRE + SUMMARY_INTERVAL, enough turns have accrued → fires again
    next_turn = _FIRST_FIRE + SUMMARY_INTERVAL
    ctx.session.events = _events(next_turn)
    with patch.object(cb, "summarize_conversation", return_value="S-NEXT"):
        cb.before_agent_callback(ctx)
    assert ctx.state["_last_summarized_turn"] == next_turn
    assert ctx.state["conversation_summary"] == "S-NEXT"


@test
def callback_empty_summary_preserves_prior():
    ctx = _make_ctx(
        state={"conversation_summary": "GOOD PRIOR SUMMARY"},
        events=_events(_FIRST_FIRE),
    )
    with patch.object(cb, "summarize_conversation", return_value=""):
        cb.before_agent_callback(ctx)
    assert ctx.state["conversation_summary"] == "GOOD PRIOR SUMMARY"
    assert "_last_summarized_turn" not in ctx.state


@test
def callback_summarizer_exception_does_not_crash():
    ctx = _make_ctx(events=_events(_FIRST_FIRE))
    with patch.object(cb, "summarize_conversation", side_effect=RuntimeError("boom")):
        # Must not raise
        cb.before_agent_callback(ctx)
    assert ctx.state.get("_summarizing") is False


# ══════════════════════════════════════════════════════════════════════════════
# 7. after_tool_callback
# ══════════════════════════════════════════════════════════════════════════════

@test
def after_tool_persists_member_profile():
    ctx = _make_ctx()
    tool = SimpleNamespace(name="get_member_summary")
    response = {"member_id": "M001", "name": "Ravi Shankar"}
    result = ta._after_tool_callback(tool, {}, ctx, response)
    assert result is None  # callback never short-circuits
    assert ctx.state["member_profile"]["name"] == "Ravi Shankar"


@test
def after_tool_ignores_unrelated_tools():
    ctx = _make_ctx()
    tool = SimpleNamespace(name="fetch_rxnorm_drug_info")
    ta._after_tool_callback(tool, {}, ctx, {"rxcui": "123"})
    assert "member_profile" not in ctx.state


@test
def after_tool_ignores_non_dict_response():
    ctx = _make_ctx()
    tool = SimpleNamespace(name="get_member_summary")
    ta._after_tool_callback(tool, {}, ctx, "oops a string")
    assert "member_profile" not in ctx.state


# ══════════════════════════════════════════════════════════════════════════════
# 8. Agent-graph wiring sanity
# ══════════════════════════════════════════════════════════════════════════════

@test
def root_agent_has_expected_name_and_subagents():
    assert root_agent.name == "HealthAssistantAgent"
    sub_names = {a.name for a in root_agent.sub_agents}
    assert sub_names == {
        "SymptomCheckerAgent",
        "DiseaseManagementAgent",
        "DietNutritionAgent",
        "MedicationInfoAgent",
    }


@test
def every_agent_has_safety_guard_installed():
    for agent in [root_agent, *root_agent.sub_agents]:
        assert agent.before_model_callback is safety_guard, (
            f"{agent.name} is missing safety_guard"
        )


@test
def root_agent_has_summarizer_callback():
    assert root_agent.before_agent_callback is cb.before_agent_callback
    assert root_agent.after_tool_callback is ta._after_tool_callback


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    passed = failed = 0
    failures: list[tuple[str, str]] = []
    for fn in _TESTS:
        name = fn.__name__
        try:
            fn()
        except Exception as e:
            failed += 1
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"[FAIL] {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"[PASS] {name}")

    total = passed + failed
    print("\n" + "-" * 60)
    print(f"  {passed}/{total} tests passed")
    if failures:
        print("  Failures:")
        for name, msg in failures:
            print(f"    - {name}: {msg}")
    print("-" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
