"""
tools/safety.py

Deterministic safety guardrail for Healix.

The LLM prompt already instructs each agent to escalate emergencies, but prompts
are probabilistic. This module adds a rule-based before_model_callback that
scans the latest user message for red-flag phrases (cardiac, stroke, severe
respiratory distress, anaphylaxis, severe bleeding, suicidal ideation, overdose)
and short-circuits the LLM with a scripted safety response.

Why deterministic:
- Regex matching is predictable and auditable.
- Fires even if the model is distracted by earlier conversation context.
- Fast (microseconds) — runs before every LLM call at negligible cost.

Integration:
    from ..tools.safety import safety_guard
    LlmAgent(..., before_model_callback=safety_guard)

All 5 agents install it so a red flag is caught no matter which agent is
currently active.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from google.adk.models import LlmResponse
from google.genai import types

logger = logging.getLogger("healix.safety")


@dataclass(frozen=True)
class RedFlag:
    category: str
    patterns: tuple[re.Pattern, ...]
    response: str


_DISCLAIMER = (
    "\n\n⚕️ I'm an AI assistant. In any medical emergency, contact emergency "
    "services immediately — do not wait for further information here."
)

# Ordered by severity of mis-triage cost: self-harm and overdose first so
# those messages always win if multiple categories match.
_RED_FLAGS: tuple[RedFlag, ...] = (
    RedFlag(
        category="suicidal_ideation",
        patterns=tuple(re.compile(p, re.IGNORECASE) for p in (
            # Explicit ideation
            r"\bkill(?:ing)?\s+my\s*self\b",
            r"\bend(?:ing)?\s+my\s+life\b",
            r"\bsuicid(?:e|al|ide)\b",
            r"\bwant(?:ed)?\s+to\s+die\b",
            r"\bno\s+reason\s+to\s+(?:live|be\s+alive)\b",
            r"\bdon'?t\s+want\s+to\s+(?:live|be\s+alive|go\s+on)\b",
            # Informal / censored variants
            r"\bwanna\s+die\b",
            r"\bwanna\s+(?:end\s+it|kill\s+my\s*self)\b",
            r"\bunaliv(?:e|ing)\b",
            r"\bkms\b",            # text-speak for "kill myself"
            # Self-harm (same escalation path — 988 covers self-harm)
            r"\bcut(?:ting)?\s+my\s*self\b",
            r"\bself[\s-]?harm(?:ing)?\b",
            r"\bhurt(?:ing)?\s+my\s*self\b",
        )),
        response=(
            "🔴 I'm really concerned by what you just shared. You don't have to go "
            "through this alone — please reach out right now:\n\n"
            "• **988** — Suicide & Crisis Lifeline (US, 24/7, call or text)\n"
            "• **911** — if you are in immediate danger\n"
            "• **Crisis Text Line** — text HOME to 741741\n\n"
            "If you're outside the US, please contact your local emergency number "
            "or a trusted person nearby. I'm here to keep talking if it helps."
        ),
    ),
    RedFlag(
        category="overdose_poisoning",
        patterns=tuple(re.compile(p, re.IGNORECASE) for p in (
            r"\bover[\s-]?dose(?:d)?\b",
            r"\btook\s+too\s+many\s+(?:pills?|tablets?|meds?|medications?|capsules?)\b",
            r"\bswallowed\s+(?:a\s+)?(?:bottle|handful)\s+of\b",
            r"\bpoisoning\b",
            r"\bingested\s+.*(?:bleach|poison|chemical|antifreeze)\b",
        )),
        response=(
            "🔴 This sounds like a possible overdose or poisoning — please act now:\n\n"
            "• **Poison Control (US): 1-800-222-1222** — 24/7 expert guidance\n"
            "• **911** — if the person is unconscious, seizing, not breathing, or "
            "hard to wake\n\n"
            "Do NOT induce vomiting unless Poison Control tells you to. Keep the "
            "medication bottle or substance container handy so responders know "
            "exactly what was taken."
        ),
    ),
    RedFlag(
        category="cardiac",
        patterns=tuple(re.compile(p, re.IGNORECASE) for p in (
            r"\bchest\s+(?:pain|pressure|tight(?:ness)?|discomfort|heaviness)\b",
            r"\bcrushing\s+(?:chest|pain)\b",
            r"\bsqueezing\s+chest\b",
            r"\bheart\s+attack\b",
            r"\bpain\s+.*(?:left\s+arm|jaw|radiating)\b",
            # Inverted phrasings: "pressure in my chest", "tightness in chest"
            r"\b(?:pressure|tight(?:ness)?|heaviness|discomfort)\s+in\s+(?:my\s+|the\s+)?chest\b",
            r"\belephant\s+(?:on|sitting\s+on)\s+(?:my\s+)?chest\b",
        )),
        response=(
            "🔴 Chest pain, pressure, or tightness can be a sign of a heart attack. "
            "Please call **911** or get to the nearest emergency room immediately — "
            "do not drive yourself. If you are not allergic and have not been told "
            "to avoid it, chew one regular (325 mg) aspirin while you wait."
        ),
    ),
    RedFlag(
        category="stroke",
        patterns=tuple(re.compile(p, re.IGNORECASE) for p in (
            r"\b(?:face|facial)\s+(?:is\s+)?droop(?:ing|ed|s)?\b",
            r"\bdroop(?:ing|ed|s)?\s+(?:face|facial)\b",
            r"\bslurr(?:ed|ing)?\s+speech\b",
            r"\bspeech\s+(?:is\s+|was\s+)?slurr(?:ed|ing)?\b",
            r"\bcan'?t\s+speak\b",
            r"\bsudden\s+.*(?:weakness|numbness|confusion)\b",
            r"\b(?:weakness|numbness)\s+.*(?:one\s+side|left\s+side|right\s+side)\b",
            r"\b(?:arm|leg|face)\s+(?:went|going|goes|is|feels)\s+numb\b",
            r"\bhaving\s+a\s+stroke\b",
            r"\bstroke\s+symptoms?\b",
        )),
        response=(
            "🔴 These sound like signs of a possible stroke (FAST: Face, Arm, "
            "Speech, Time). Every minute matters. Call **911** immediately and "
            "note the exact time symptoms started — the ER needs it to decide on "
            "clot-busting treatment."
        ),
    ),
    RedFlag(
        category="respiratory",
        patterns=tuple(re.compile(p, re.IGNORECASE) for p in (
            r"\bcan'?t\s+breathe\b",
            r"\bunable\s+to\s+breathe\b",
            r"\bcan'?t\s+catch\s+my\s+breath\b",
            r"\bstruggl(?:e|ing)\s+to\s+breathe\b",
            r"\bchoking\b",
            r"\bgasping\s+for\s+(?:air|breath)\b",
            r"\b(?:lips|face|fingers)\s+.*(?:blue|bluish|cyanotic)\b",
            r"\bturning\s+blue\b",
        )),
        response=(
            "🔴 Severe breathing difficulty is a medical emergency. Call **911** "
            "right now. If the person is choking and conscious, use the Heimlich "
            "maneuver; if unconscious and not breathing, start CPR and stay on "
            "the line with the 911 dispatcher."
        ),
    ),
    RedFlag(
        category="anaphylaxis",
        patterns=tuple(re.compile(p, re.IGNORECASE) for p in (
            r"\bthroat\s+.*(?:closing|swelling|tight)\b",
            r"\btongue\s+.*swelling\b",
            r"\banaphyla",
            r"\bsevere\s+allergic\s+reaction\b",
        )),
        response=(
            "🔴 This could be anaphylaxis. Call **911** now. If an epinephrine "
            "auto-injector (EpiPen) is available, use it immediately in the outer "
            "thigh — do not wait to see if symptoms improve."
        ),
    ),
    RedFlag(
        category="severe_bleeding",
        patterns=tuple(re.compile(p, re.IGNORECASE) for p in (
            r"\buncontrolled\s+bleeding\b",
            r"\bbleeding\s+(?:that\s+)?won'?t\s+stop\b",
            r"\bgunshot\b",
            r"\b(?:stabbed|stab\s+wound)\b",
            r"\bheavy\s+bleeding\b",
        )),
        response=(
            "🔴 Severe bleeding is a medical emergency. Call **911** immediately. "
            "Apply firm, direct pressure to the wound with a clean cloth and do "
            "not remove any embedded object."
        ),
    ),
)


def scan_for_red_flags(text: str) -> RedFlag | None:
    """Return the first matching RedFlag for `text`, or None."""
    if not text:
        return None
    for flag in _RED_FLAGS:
        for pat in flag.patterns:
            if pat.search(text):
                return flag
    return None


def _extract_last_user_text(contents) -> str:
    """Concatenate text parts of the most recent user message in an LlmRequest."""
    if not contents:
        return ""
    for content in reversed(list(contents)):
        if getattr(content, "role", None) != "user":
            continue
        parts = getattr(content, "parts", None) or []
        texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
        if texts:
            return " ".join(texts).strip()
    return ""


def safety_guard(callback_context, llm_request):
    """
    before_model_callback: scan the latest user message for red flags.

    If a red flag is detected, short-circuit the LLM call by returning an
    LlmResponse with a scripted safety message. Otherwise return None and let
    the agent proceed normally.

    Also records the matched category in session state under '_last_red_flag'
    so downstream code / dashboards can audit escalations.
    """
    try:
        user_text = _extract_last_user_text(getattr(llm_request, "contents", None))
        flag = scan_for_red_flags(user_text)
        if flag is None:
            return None

        logger.warning(
            "Red-flag detected: category=%s agent=%s",
            flag.category,
            getattr(callback_context, "agent_name", "?"),
        )
        try:
            callback_context.state["_last_red_flag"] = flag.category
        except Exception:
            pass

        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=flag.response + _DISCLAIMER)],
            ),
        )
    except Exception:
        logger.exception("safety_guard failed — falling through to model")
        return None
