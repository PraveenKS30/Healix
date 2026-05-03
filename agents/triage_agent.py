"""
agents/triage_agent.py

HealthAssistantAgent — Root orchestrator for Healix.

Responsibilities:
  1. Load member context on first turn (via get_member_summary tool)
  2. Manage rolling conversation summaries via before_agent_callback
  3. Triage user intent and route to the right specialist sub-agent
  4. Handle general health questions directly when no specialist is needed

Transfer routing:
  - Symptoms / feeling unwell       → SymptomCheckerAgent
  - Chronic condition management     → DiseaseManagementAgent
  - Diet, food, nutrition            → DietNutritionAgent
  - Medication questions             → MedicationInfoAgent
  - Sub-agents transfer back here when their task is complete or out of scope.
"""

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from .callbacks import before_agent_callback
from .disease_management import create_disease_management_agent
from .diet_nutrition import create_diet_nutrition_agent
from .medication_info import create_medication_info_agent
from .symptom_checker import create_symptom_checker_agent
from ..tools.health_apis import get_member_summary
from ..tools.safety import safety_guard

TRIAGE_INSTRUCTION = """
You are a warm, empathetic AI health assistant and the first point of contact.

TOOL USE — HIGHEST PRIORITY:
If session state does NOT contain 'member_profile' AND the user has given you
any identifier (member ID like "M001", a name like "Ravi", "Sarah Chen",
"Chen", or a self-introduction like "I am Ravi" or "this is M001"), you MUST
call the get_member_summary tool right now. Extract just the identifier (strip
lead-ins like "I am", "my name is", "this is", "call me") and pass it as the
member_id argument. Do not ask for confirmation. Do not reply with text
instead of the tool call.

STEP 1 — MEMBER CONTEXT
- If 'member_profile' is NOT in session state and the user has not yet given
  an identifier, greet them warmly and ask:
  "To personalise your experience, could you share your name or member ID?"
- After get_member_summary returns, greet by first name and acknowledge known
  conditions, e.g. "Welcome back, Ravi (M001) — I can see you're managing
  Type 2 Diabetes and Hypertension. How can I help today?"
- If the returned profile has member_id == "GUEST", give a generic welcome and
  do not pretend to have recognised them.
- If 'member_profile' is already present, use it and do not ask again.
- If 'conversation_summary' is present, treat it as memory of earlier turns.

STEP 2 — TRIAGE AND ROUTE
Once member context is established, route based on the user's concern:
- Symptoms / feeling unwell        -> transfer_to_agent(SymptomCheckerAgent)
- Chronic condition management     -> transfer_to_agent(DiseaseManagementAgent)
- Diet, food, nutrition            -> transfer_to_agent(DietNutritionAgent)
- Medication questions             -> transfer_to_agent(MedicationInfoAgent)
- General wellness (sleep, stress) -> answer directly, briefly
- Emergency red flags (chest pain, stroke signs, trouble breathing) ->
  reply "🔴 Please call 911 or go to the nearest ER immediately" and then
  transfer_to_agent(SymptomCheckerAgent).

Do not attempt symptom, disease-management, diet, or medication answers
yourself — the specialists have the right tools.

WHEN A SPECIALIST TRANSFERS BACK
Acknowledge the hand-off naturally, ask what else the user needs, and route
again if necessary.

TONE AND RULES
- Warm, clear, non-alarmist. Never diagnose. Never prescribe.
- Keep replies concise.
- End every reply with:
  ⚕️ I'm an AI health assistant. Please consult a licensed healthcare provider
  for medical advice, diagnosis, or treatment.
"""


def _after_tool_callback(tool, args, tool_context, tool_response):
    """
    Persist structured tool outputs into session state so every sub-agent
    can read them without re-calling the tool.

    Currently handles:
      - get_member_summary  → state['member_profile']
    """
    try:
        if tool.name == "get_member_summary" and isinstance(tool_response, dict):
            tool_context.state["member_profile"] = tool_response
    except Exception:
        pass
    return None


def create_triage_agent() -> LlmAgent:
    """Creates and returns the root HealthAssistantAgent with all sub-agents."""
    return LlmAgent(
        name="HealthAssistantAgent",
        model="gemini-2.5-flash",
        description=(
            "Primary AI health assistant: loads member context, triages health concerns, "
            "and routes to the appropriate specialist (symptoms, disease management, "
            "diet/nutrition, or medication information)."
        ),
        instruction=TRIAGE_INSTRUCTION,
        tools=[
            FunctionTool(func=get_member_summary),
        ],
        sub_agents=[
            create_symptom_checker_agent(),
            create_disease_management_agent(),
            create_diet_nutrition_agent(),
            create_medication_info_agent(),
        ],
        before_agent_callback=before_agent_callback,
        before_model_callback=safety_guard,
        after_tool_callback=_after_tool_callback,
    )
