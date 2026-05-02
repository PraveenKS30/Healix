# Healix 🏥
### A Virtual Clinical Care Agent

> A production-ready, multi-agent AI health assistant built on **Google ADK** and **Gemini 2.5 Flash** — with deterministic safety guardrails, rolling conversation summarization, and five free public health APIs wired in out of the box.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Google ADK](https://img.shields.io/badge/Google%20ADK-1.x-orange.svg)](https://google.github.io/adk-docs/)
[![Gemini 2.5 Flash](https://img.shields.io/badge/Gemini-2.5%20Flash-green.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [What Is Healix?](#what-is-healix)
2. [Architecture](#architecture)
3. [Key Features](#key-features)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Running the App](#running-the-app)
8. [Project Structure](#project-structure)
9. [Agents](#agents)
10. [Health API Integrations](#health-api-integrations)
11. [Safety Guardrails](#safety-guardrails)
12. [Rolling Conversation Summarization](#rolling-conversation-summarization)
13. [Test Member Profiles](#test-member-profiles)
14. [Running Tests and Diagnostics](#running-tests-and-diagnostics)
15. [Extending the Project](#extending-the-project)
16. [Important Disclaimer](#important-disclaimer)

---

## What Is Healix?

Healix is a reference implementation of a multi-agent virtual clinical care assistant. A user identifies themselves by name or member ID, and the root **triage agent** loads their health profile, then routes their query to the appropriate specialist agent — symptoms, chronic disease management, diet/nutrition, or medication information.

Each specialist:
- Pulls context from the shared session state (member profile, prior conversation summary)
- Calls the relevant free public health APIs (MedlinePlus, PubMed, SNOMED CT, RxNorm, OpenFDA)
- Personalizes every response around the member's known conditions, medications, and allergies
- Returns to the triage agent when its task is complete

A **deterministic regex-based safety layer** runs before every single LLM call across all five agents — cardiac events, stroke, respiratory distress, anaphylaxis, overdose, severe bleeding, and suicidal ideation all short-circuit the model and return a scripted 911/988 escalation regardless of conversation context.

---

## Architecture

```
User
 │
 ▼
HealthAssistantAgent  (root — triage + member context)
 │   before_agent_callback → rolling summarizer  (all agents)
 │   before_model_callback → safety guardrail     (all agents)
 │
 ├─── transfer ──► SymptomCheckerAgent
 │                    tools: MedlinePlus, PubMed
 │
 ├─── transfer ──► DiseaseManagementAgent
 │                    tools: MedlinePlus, PubMed, SNOMED CT
 │
 ├─── transfer ──► DietNutritionAgent
 │                    tools: MedlinePlus
 │
 └─── transfer ──► MedicationInfoAgent
                      tools: RxNorm, OpenFDA interactions, MedlinePlus

Session State (shared across all agents)
  member_profile          ← loaded once by root agent via get_member_summary
  conversation_summary    ← rolling summary written by before_agent_callback
  _last_summarized_turn   ← tracks interval gating
  _last_red_flag          ← audit trail for safety escalations
```

Agents transfer **back** to `HealthAssistantAgent` when their task is done, keeping the conversation flowing naturally.

---

## Key Features

| Feature | Detail |
|---|---|
| **Multi-agent routing** | 5 specialist agents; root agent triages and transfers |
| **Persistent sessions** | SQLite via ADK `DatabaseSessionService`; history survives restarts |
| **Member context** | Simulated EHR lookup by member ID or name; 4 test profiles included |
| **Deterministic safety** | Regex guardrail runs before every LLM call; 7 emergency categories |
| **Rolling summarization** | Compresses conversation history after `MAX_TURNS`; re-fires on `SUMMARY_INTERVAL` |
| **RxNorm canonicalization** | Resolves brand names and dosage-prefixed strings to generic ingredients before interaction checks |
| **5 free health APIs** | MedlinePlus, PubMed, SNOMED CT, RxNorm, OpenFDA — no API keys required |
| **ADK Dev UI** | Full `adk web` support with event timeline and state panel |
| **Windows-safe** | UTF-8 stdout reconfiguration prevents cp1252 crashes on emoji-heavy model output |

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or higher |
| Google ADK | 1.x (`google-adk`) |
| Gemini API key | [Get one free at ai.google.dev](https://ai.google.dev/) |
| Internet access | Required for health API calls at runtime |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/healix.git
cd healix
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

### 1. Create your `.env` file

Copy the example and add your credentials:

```bash
cp .env.example healix/.env
```

Then open `healix/.env` and fill in your key:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

> **Using Vertex AI instead of AI Studio?** Set these instead:
> ```env
> GOOGLE_GENAI_USE_VERTEXAI=true
> GOOGLE_CLOUD_PROJECT=your-project-id
> GOOGLE_CLOUD_LOCATION=us-central1
> ```
> Authentication uses Application Default Credentials (`gcloud auth application-default login`).

### 2. (Optional) Tune the summarization thresholds

Open `healix/tools/conversation_utils.py` and adjust:

```python
MAX_TURNS = 20          # Summarization starts firing after this many turns
SUMMARY_INTERVAL = 10   # Re-summarize only every N new turns past MAX_TURNS
```

Lowering `MAX_TURNS` to `5` during development lets you trigger and inspect summarization quickly.

### 3. (Optional) Change the session database path

The default SQLite file is `healix.db` in your working directory. Override it:

```env
HEALIX_DB_URL=sqlite+aiosqlite:///path/to/your.db
```

---

## Running the App

### ADK Web UI (recommended for development)

From the **repo root** (the directory that *contains* the `healix/` folder):

```bash
adk web
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

1. Select **healix** from the app dropdown
2. Type `Hello` — the agent will ask for your name or member ID
3. Type one of the test IDs: `M001`, `M002`, `M003`, or `M004`
4. The root agent loads your profile and greets you by name
5. Describe a health concern and watch the triage and specialist agents work

**Inspecting session state:**
- Click the **State** tab in the right panel to see `member_profile`, `conversation_summary`, and safety escalation flags live.
- Watch the terminal for summarization log lines: `healix.triage INFO: Summarization firing: turns=21, ...`

### Programmatic usage

```python
import asyncio
from google.adk.runners import InMemoryRunner
from google.genai import types
from healix import root_agent

async def chat():
    runner = InMemoryRunner(agent=root_agent, app_name="Healix")
    session = await runner.session_service.create_session(
        app_name="Healix", user_id="M001"
    )

    for text in ["Hello", "M001", "I have been having chest pain"]:
        content = types.Content(role="user", parts=[types.Part(text=text)])
        async for event in runner.run_async(
            user_id="M001", session_id=session.id, new_message=content
        ):
            if event.content and event.content.parts:
                for p in event.content.parts:
                    if getattr(p, "text", None):
                        print(f"[{event.author}] {p.text[:300]}")

asyncio.run(chat())
```

---

## Project Structure

```
healix-repo/                          ← repo root (run `adk web` from here)
│
├── healix/                           ← Python package (discovered by adk web)
│   ├── __init__.py                   ← exposes root_agent + session_service
│   ├── session.py                    ← SQLite session persistence
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── callbacks.py              ← shared before_agent_callback (summarizer)
│   │   ├── triage_agent.py           ← root HealthAssistantAgent
│   │   ├── symptom_checker.py        ← SymptomCheckerAgent
│   │   ├── disease_management.py     ← DiseaseManagementAgent
│   │   ├── diet_nutrition.py         ← DietNutritionAgent
│   │   └── medication_info.py        ← MedicationInfoAgent
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── health_apis.py            ← member lookup + 5 health API wrappers
│   │   ├── safety.py                 ← deterministic red-flag guardrail
│   │   └── conversation_utils.py     ← rolling summarizer + turn counter
│   │
│   └── tests/
│       ├── test_scenarios.py         ← 34-test offline test suite
│       ├── diagnose_summarization.py ← offline summarizer trigger verification
│       └── diagnose_member_flow.py   ← end-to-end ADK runner diagnostic
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
└── LICENSE
```

---

## Agents

### HealthAssistantAgent (root / triage)

The entry point for every conversation. Responsibilities:
- Ask for the member's name or ID on the first turn
- Call `get_member_summary` to load the health profile into session state
- Route the user's concern to the right specialist
- Handle general wellness questions (sleep, stress) directly
- Greet returning users and acknowledge prior conversation summary
- **Emergency fast path**: replies `🔴 Please call 911...` before transferring to SymptomCheckerAgent for chest pain, stroke signs, severe breathing difficulty

### SymptomCheckerAgent

Guides users through a structured symptom assessment:
- Collects location, duration, severity (1–10), character, associated symptoms, onset
- Cross-references with known conditions from the member profile
- Calls `fetch_medlineplus_guidelines` for evidence-based info
- Classifies urgency: 🔴 Emergency / 🟡 Urgent / 🟢 Manage at home
- Provides self-care tips, flagging anything unsafe given the member's medications/allergies

### DiseaseManagementAgent

Chronic condition specialist covering diabetes, hypertension, COPD, CAD, thyroid, dyslipidemia, and more:
- Calls `fetch_medlineplus_guidelines` and `fetch_pubmed_guidelines`
- Uses `fetch_snomed_code` for clinical coding confirmation
- Covers medication adherence, target monitoring ranges, lifestyle modification, red flags, and follow-up schedules
- Cross-references all advice against the member's current medications for conflicts

### DietNutritionAgent

Condition-specific dietary guidance:
- Tailored to the member's known conditions (DASH diet for hypertension, low-GI for diabetes, etc.)
- Structured output: foods to include / avoid, hydration, meal timing, practical swaps
- Flags allergy conflicts from the member profile
- Calls `fetch_medlineplus_guidelines` for evidence base

### MedicationInfoAgent

Drug information and interaction safety:
- `fetch_rxnorm_drug_info` — looks up RxNorm drug details
- `fetch_drug_interactions` — checks the queried drug against all current medications via OpenFDA labels; resolves brand names and dosage strings to generic names via RxNorm before the scan
- `fetch_medlineplus_guidelines` — patient-level drug education
- Flags any allergy class conflict from the member profile with a prominent `⚠️ ALLERGY WARNING`

---

## Health API Integrations

All APIs are **free and require no authentication**. They are called at runtime via `httpx`.

| API | Endpoint | Used By | Purpose |
|---|---|---|---|
| **NLM MedlinePlus** | `connect.medlineplus.gov/service` | All specialists | Evidence-based patient guidelines |
| **PubMed E-utilities** | `eutils.ncbi.nlm.nih.gov` | Symptom, Disease | Recent clinical abstracts |
| **SNOMED CT (NLM Snowstorm)** | `browser.ihtsdotools.org/snowstorm` | Disease | Clinical coding lookup |
| **RxNorm (NLM)** | `rxnav.nlm.nih.gov/REST/drugs.json` | Medication | Drug lookup + name canonicalization |
| **OpenFDA Drug Labels** | `api.fda.gov/drug/label.json` | Medication | Drug interaction text scan |

> **Note**: The NLM RxNav interaction endpoint was deprecated in January 2024. Healix uses OpenFDA drug labels as a best-effort replacement and always surfaces a disclaimer to confirm with a pharmacist.

---

## Safety Guardrails

A **deterministic regex-based** `before_model_callback` runs before every LLM call on all five agents. It cannot be confused by conversation context, model distraction, or prompt injection.

| Category | Example Triggers | Response |
|---|---|---|
| `suicidal_ideation` | "I want to die", "wanna end it", "kms", "cutting myself", "unalive" | 988 Lifeline + 911 escalation |
| `overdose_poisoning` | "I overdosed", "took too many pills", "swallowed a bottle of" | Poison Control (1-800-222-1222) + 911 |
| `cardiac` | "chest pain", "pressure in my chest", "elephant on my chest", "heart attack" | 911 + aspirin guidance |
| `stroke` | "face drooping", "arm went numb", "slurred speech", "sudden weakness" | 911 + FAST protocol + note symptom start time |
| `respiratory` | "can't breathe", "struggling to breathe", "gasping for air", "turning blue" | 911 + CPR/Heimlich guidance |
| `anaphylaxis` | "throat closing", "tongue swelling", "severe allergic reaction" | 911 + EpiPen instruction |
| `severe_bleeding` | "uncontrolled bleeding", "bleeding won't stop", "gunshot", "stabbed" | 911 + direct pressure instruction |

When a red flag fires:
1. The LLM call is **skipped entirely** — the scripted response is returned directly
2. The matched category is written to `session_state["_last_red_flag"]` for audit
3. A `WARNING` log line is emitted: `healix.safety WARNING: Red-flag detected: category=cardiac agent=SymptomCheckerAgent`

---

## Rolling Conversation Summarization

Long conversations are compressed to keep each LLM call small and cheap.

**How it works:**

1. `before_agent_callback` fires before each agent invocation (root + all sub-agents)
2. It counts the total user+model turns in the session
3. If `turns > MAX_TURNS` **and** `turns - last_summarized >= SUMMARY_INTERVAL`, it calls `summarize_conversation(events)`
4. The summary is stored in `session_state["conversation_summary"]`; every agent prompt instructs the LLM to treat it as memory of prior turns
5. If the summarizer returns empty (e.g., transient API failure), the prior summary is kept — no overwrite

**Default thresholds** (adjust in `conversation_utils.py`):

```
MAX_TURNS = 20       → first summarization at turn 21
SUMMARY_INTERVAL = 10 → re-summarizes at turns 31, 41, 51, ...
```

**Observing summarization:**

```
# Terminal (adk web logs):
healix.triage INFO: Summarization firing: turns=21, last_summarized=0, interval=10
healix.triage INFO: Summary stored (842 chars): The user identified as Ravi Shankar (M001)...

# ADK Web UI → State tab:
conversation_summary: "The user identified as Ravi Shankar..."
_last_summarized_turn: 21
```

**Offline verification** (no API calls, no tokens spent):

```bash
python healix/tests/diagnose_summarization.py
```

---

## Test Member Profiles

Four simulated EHR profiles are included in `tools/health_apis.py`. Use these IDs in the chat or tests:

| ID | Name | Age | Conditions | Medications | Allergies |
|---|---|---|---|---|---|
| **M001** | Ravi Shankar | 52 M | Type 2 Diabetes, Hypertension | Metformin 500mg BD, Amlodipine 5mg OD | Penicillin |
| **M002** | Sarah Chen | 34 F | Hypothyroidism, Generalized Anxiety Disorder | Levothyroxine 50mcg OD | None |
| **M003** | James Wilson | 67 M | COPD, Dyslipidemia, Coronary Artery Disease | Atorvastatin 40mg, Metoprolol 25mg, Aspirin 81mg | Sulfa drugs |
| **M004** | Priya Patel | 28 F | None | None | Shellfish |

Any unrecognised name or ID returns a **GUEST** profile so the conversation can continue gracefully.

---

## Running Tests and Diagnostics

### Full offline test suite (34 tests, no API calls)

```bash
python healix/tests/test_scenarios.py
```

Covers: safety guardrail (8), drug canonicalization (6), interaction guard (2), member lookup (4), conversation utils (3), before_agent_callback state machine (5), after_tool_callback (3), wiring sanity (3).

### Summarization trigger verification (offline, no tokens)

```bash
python healix/tests/diagnose_summarization.py
```

Drives the callback through 1–43 simulated turns with a mocked summarizer and prints a per-turn table showing exactly when triggers fire.

### End-to-end member flow diagnostic (requires `GOOGLE_API_KEY`)

```bash
python healix/tests/diagnose_member_flow.py
```

Runs two real turns (`Hello` → `M001`) through an `InMemoryRunner`, prints every event (tool call, tool result, model text), and dumps final session state. Use this to verify member lookup works end-to-end after any prompt or tool changes.

---

## Extending the Project

### Add a new member profile

Open `healix/tools/health_apis.py` and add an entry to `_MEMBER_PROFILES`:

```python
"M005": {
    "member_id": "M005",
    "name": "Alex Rivera",
    "age": 45,
    "gender": "Non-binary",
    "known_conditions": ["Asthma", "Chronic Migraine"],
    "current_medications": ["Albuterol inhaler PRN", "Topiramate 50mg once daily"],
    "allergies": ["Aspirin"],
    "primary_care_physician": "Dr. Jane Kim",
},
```

### Add a new specialist agent

1. Create `healix/agents/your_specialist.py` following the pattern in `symptom_checker.py`
2. Import and wire `before_agent_callback` and `safety_guard`
3. Add the agent to `sub_agents=[...]` in `create_triage_agent()` in `triage_agent.py`
4. Add a routing rule in `TRIAGE_INSTRUCTION`

### Add a new health API tool

1. Write the function in `healix/tools/health_apis.py`
2. Export it from `healix/tools/__init__.py`
3. Add it as a `FunctionTool(func=your_function)` to the relevant agent

### Replace the simulated EHR with a real one

Swap the `_MEMBER_PROFILES` dict in `get_member_summary` with a real database or REST call. The function signature and return shape are the same — the agents see no difference.

### Connect to a real database for sessions

Update `HEALIX_DB_URL` in your `.env`:

```env
# PostgreSQL example
HEALIX_DB_URL=postgresql+asyncpg://user:password@host:5432/healix
```

ADK's `DatabaseSessionService` supports any SQLAlchemy-compatible async URL.

---

## Important Disclaimer

> **Healix is a demonstration project for educational and research purposes only.**
>
> It is **not** a licensed medical device, clinical decision support system, or healthcare product. It does not provide medical diagnoses, treatment recommendations, or professional medical advice.
>
> - Do **not** use it as a substitute for professional medical consultation.
> - Do **not** deploy it in a production healthcare setting without appropriate clinical oversight, regulatory approval, and compliance review (HIPAA, GDPR, MDR, etc.).
> - The safety guardrails are rule-based and provided as a safety net — they are not a substitute for proper clinical triage.
> - The test member profiles contain synthetic, fictional health data.
>
> Always consult a licensed healthcare provider for any medical concerns.

---

## License

This project is released under the [MIT License](LICENSE).

---

*Built with [Google ADK](https://google.github.io/adk-docs/) · Powered by [Gemini 2.5 Flash](https://ai.google.dev/)*

