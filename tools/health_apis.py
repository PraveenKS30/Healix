"""
tools/health_apis.py

Public health API wrappers and member context lookup for Healix.
All APIs are free and require no authentication.
"""

import httpx


# ── Member Context (simulated EHR lookup) ────────────────────────────────────

_MEMBER_PROFILES = {
    "M001": {
        "member_id": "M001",
        "name": "Ravi Shankar",
        "age": 52,
        "gender": "Male",
        "known_conditions": ["Type 2 Diabetes Mellitus", "Hypertension"],
        "current_medications": ["Metformin 500mg twice daily", "Amlodipine 5mg once daily"],
        "allergies": ["Penicillin"],
        "primary_care_physician": "Dr. Anjali Mehta",
    },
    "M002": {
        "member_id": "M002",
        "name": "Sarah Chen",
        "age": 34,
        "gender": "Female",
        "known_conditions": ["Hypothyroidism", "Generalized Anxiety Disorder"],
        "current_medications": ["Levothyroxine 50mcg once daily"],
        "allergies": [],
        "primary_care_physician": "Dr. Brian Foster",
    },
    "M003": {
        "member_id": "M003",
        "name": "James Wilson",
        "age": 67,
        "gender": "Male",
        "known_conditions": ["COPD", "Dyslipidemia", "Coronary Artery Disease"],
        "current_medications": [
            "Atorvastatin 40mg once daily",
            "Metoprolol Succinate 25mg once daily",
            "Aspirin 81mg once daily",
        ],
        "allergies": ["Sulfa drugs"],
        "primary_care_physician": "Dr. Maria Santos",
    },
    "M004": {
        "member_id": "M004",
        "name": "Priya Patel",
        "age": 28,
        "gender": "Female",
        "known_conditions": [],
        "current_medications": [],
        "allergies": ["Shellfish"],
        "primary_care_physician": "Dr. Kevin Nguyen",
    },
}

_GENERIC_PROFILE = {
    "member_id": "GUEST",
    "name": "Guest User",
    "age": None,
    "gender": "Not specified",
    "known_conditions": [],
    "current_medications": [],
    "allergies": [],
    "primary_care_physician": "Not on file",
}


def get_member_summary(member_id: str) -> dict:
    """
    Look up a member's health profile by member ID or partial name.

    Args:
        member_id: Member ID (e.g. "M001") or partial name (e.g. "Ravi")

    Returns:
        Member health profile dict with name, age, gender, conditions,
        medications, allergies, and primary care physician.
        Returns a generic guest profile if no match is found.
    """
    # Exact ID match
    key = member_id.strip().upper()
    if key in _MEMBER_PROFILES:
        return _MEMBER_PROFILES[key]

    # Partial name match (case-insensitive)
    query = member_id.strip().lower()
    for profile in _MEMBER_PROFILES.values():
        if query in profile["name"].lower():
            return profile

    # No match — return generic profile so the agent can still proceed
    guest = _GENERIC_PROFILE.copy()
    guest["name"] = member_id.strip() or "Guest User"
    return guest


# ── NLM MedlinePlus Connect ───────────────────────────────────────────────────

def fetch_medlineplus_guidelines(diagnosis: str) -> dict:
    """
    Fetch clinical summary and guidelines for a diagnosis from MedlinePlus.

    Args:
        diagnosis: Plain-text diagnosis (e.g. 'Type 2 Diabetes')

    Returns:
        Dictionary with guidelines list (title, summary, url) or error.
    """
    url = "https://connect.medlineplus.gov/service"
    params = {
        "mainSearchCriteria.v.cs": "2.16.840.1.113883.6.103",
        "mainSearchCriteria.v.dn": diagnosis,
        "informationRecipient.languageCode.c": "en",
        "knowledgeResponseType": "application/json",
    }
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        entries = data.get("feed", {}).get("entry", [])
        results = []
        for entry in entries[:3]:
            results.append({
                "title": entry.get("title", {}).get("_value", ""),
                "summary": entry.get("summary", {}).get("_value", "")[:500],
                "url": entry.get("link", [{}])[0].get("href", ""),
            })
        return {"source": "NLM MedlinePlus", "diagnosis": diagnosis, "guidelines": results}
    except Exception as e:
        return {"source": "NLM MedlinePlus", "diagnosis": diagnosis, "error": str(e)}


# ── PubMed (NCBI E-utilities) ─────────────────────────────────────────────────

def fetch_pubmed_guidelines(diagnosis: str, max_results: int = 3) -> dict:
    """
    Fetch recent PubMed abstracts on clinical guidelines for a diagnosis.

    Args:
        diagnosis: Plain-text diagnosis
        max_results: Number of abstracts to fetch (default 3)

    Returns:
        Dictionary with PMIDs and abstract text, or error.
    """
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    query = f"{diagnosis} clinical guidelines treatment"

    try:
        with httpx.Client(timeout=15) as client:
            search_resp = client.get(search_url, params={
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance",
            })
            search_resp.raise_for_status()
            pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])

            if not pmids:
                return {"source": "PubMed", "diagnosis": diagnosis, "abstracts": []}

            fetch_resp = client.get(fetch_url, params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "rettype": "abstract",
                "retmode": "text",
            })
            fetch_resp.raise_for_status()
            abstract_text = fetch_resp.text

        return {
            "source": "PubMed NCBI",
            "diagnosis": diagnosis,
            "pmids": pmids,
            "abstracts_text": abstract_text[:2000],
        }
    except Exception as e:
        return {"source": "PubMed", "diagnosis": diagnosis, "error": str(e)}


# ── SNOMED CT ─────────────────────────────────────────────────────────────────

def fetch_snomed_code(term: str) -> dict:
    """
    Look up SNOMED CT concept ID for a clinical term.

    Args:
        term: Clinical term (diagnosis, finding, procedure)

    Returns:
        Dictionary with SNOMED concept ID, preferred term, and active status.
    """
    url = "https://browser.ihtsdotools.org/snowstorm/snomed-ct/browser/MAIN/descriptions"
    params = {
        "term": term,
        "active": "true",
        "language": "english",
        "type": "900000000000003001",
        "offset": 0,
        "limit": 3,
    }
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        items = data.get("items", [])
        results = []
        for item in items:
            concept = item.get("concept", {})
            results.append({
                "snomed_id": concept.get("conceptId"),
                "preferred_term": concept.get("pt", {}).get("term", term),
                "active": concept.get("active"),
            })
        return {"source": "SNOMED CT (NLM)", "term": term, "results": results}
    except Exception as e:
        return {"source": "SNOMED CT", "term": term, "error": str(e)}


# ── RxNorm Drug Lookup ────────────────────────────────────────────────────────

def fetch_rxnorm_drug_info(drug_name: str) -> dict:
    """
    Look up drug information from RxNorm (NLM).

    Args:
        drug_name: Medication name

    Returns:
        Dictionary with RxCUI, drug name, synonym, term type, or error.
    """
    url = "https://rxnav.nlm.nih.gov/REST/drugs.json"
    params = {"name": drug_name}
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        concept_groups = data.get("drugGroup", {}).get("conceptGroup", [])
        results = []
        for group in concept_groups:
            for prop in group.get("conceptProperties", []):
                results.append({
                    "rxcui": prop.get("rxcui"),
                    "name": prop.get("name"),
                    "synonym": prop.get("synonym"),
                    "tty": prop.get("tty"),
                })
        return {"source": "RxNorm (NLM)", "drug": drug_name, "results": results[:3]}
    except Exception as e:
        return {"source": "RxNorm", "drug": drug_name, "error": str(e)}


# ── Drug Interaction Checker ──────────────────────────────────────────────────
#
# NOTE: The NLM RxNav interaction endpoint was deprecated in January 2024, so
# we now use OpenFDA's drug-label API as a best-effort replacement. For each
# drug we fetch its FDA label and search the drug_interactions / contraindications
# free-text for mentions of the other drugs. This catches interactions that FDA
# labeling explicitly warns about, but is NOT a comprehensive pharmacy-grade
# DDI service — always advise the user to confirm with a pharmacist.

_FDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
_RXNORM_DRUGS_URL = "https://rxnav.nlm.nih.gov/REST/drugs.json"
_FDA_INTERACTION_DISCLAIMER = (
    "Best-effort label-text search via OpenFDA. Not a comprehensive "
    "drug-interaction service. Please confirm with a licensed pharmacist."
)

# Words that masquerade as the leading token but aren't the drug name.
_DOSAGE_NOISE = frozenset({
    "extended-release", "sustained-release", "immediate-release",
    "controlled-release", "delayed-release", "long-acting", "short-acting",
    "er", "xr", "sr", "cr", "ir", "xl", "la", "dr",
    "oral", "injectable", "topical", "inhaled", "generic", "brand",
})


def _drug_base_fallback(name: str) -> str:
    """
    Offline drug-name normalizer used when RxNorm is unreachable.

    Picks the first token that isn't a dosage prefix, dose string, or
    numeric/unit fragment. 'Sustained-release metformin 500mg' → 'metformin'.
    """
    if not name:
        return ""
    for raw in name.lower().split():
        token = raw.strip(",.()[]")
        if not token or token in _DOSAGE_NOISE:
            continue
        if token[0].isdigit():
            continue
        if token.endswith(("mg", "mcg", "g", "ml", "%")):
            continue
        return token
    # Last resort: first alphabetic token
    for raw in name.lower().split():
        if raw and raw[0].isalpha():
            return raw.strip(",.()[]")
    return ""


# Preference order for RxNorm term-type (tty):
#   IN = ingredient (generic, what we want)
#   MIN = multi-ingredient
#   PIN = precise ingredient
#   BN = brand name (fallback)
_RXNORM_TTY_PREF = ("IN", "MIN", "PIN", "BN")


def _canonical_drug_name(client: httpx.Client, drug_name: str) -> str:
    """
    Resolve a free-form drug string to its canonical generic name via RxNorm.

    'Metformin 500mg twice daily'        → 'metformin'
    'Sustained-release metformin 500mg'  → 'metformin'  (RxNorm handles this)
    'Amoxil'                             → 'amoxicillin' (brand → ingredient)

    Falls back to _drug_base_fallback on lookup failure so downstream code
    always receives a non-empty normalized string when one can be inferred.
    """
    if not drug_name:
        return ""
    try:
        resp = client.get(_RXNORM_DRUGS_URL, params={"name": drug_name}, timeout=10)
        resp.raise_for_status()
        groups = resp.json().get("drugGroup", {}).get("conceptGroup", []) or []
        for tty in _RXNORM_TTY_PREF:
            for group in groups:
                if group.get("tty") != tty:
                    continue
                for prop in group.get("conceptProperties") or []:
                    name = (prop.get("name") or "").strip().lower()
                    if name:
                        return name.split()[0]
    except Exception:
        pass
    return _drug_base_fallback(drug_name)


def _fetch_label(client: httpx.Client, drug_base: str) -> dict | None:
    """Fetch the FDA drug label for a drug name, or None if not found."""
    try:
        resp = client.get(
            _FDA_LABEL_URL,
            params={
                "search": (
                    f'openfda.generic_name:"{drug_base}" '
                    f'OR openfda.brand_name:"{drug_base}" '
                    f'OR openfda.substance_name:"{drug_base}"'
                ),
                "limit": 1,
            },
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        results = resp.json().get("results") or []
        return results[0] if results else None
    except Exception:
        return None


def _interaction_text(label: dict) -> str:
    """Flatten the label's interaction-relevant fields into a single lowercase string."""
    fields = (
        "drug_interactions",
        "drug_and_or_laboratory_test_interactions",
        "contraindications",
        "warnings_and_cautions",
        "boxed_warning",
    )
    parts: list[str] = []
    for f in fields:
        vals = label.get(f)
        if isinstance(vals, list):
            parts.extend(str(v) for v in vals if v)
        elif isinstance(vals, str):
            parts.append(vals)
    return " ".join(parts).lower()


def fetch_drug_interactions(drug_names: list[str]) -> dict:
    """
    Check for potential drug interactions using OpenFDA drug labels.

    For each drug in the list, the FDA label is fetched and its drug-interaction
    / contraindication / warnings text is scanned for mentions of the other
    drug names. Matches are reported with a short context snippet.

    Args:
        drug_names: List of medication name strings
                    (e.g. ["Metformin 500mg", "Ibuprofen"]).
                    Each name is normalized to its RxNorm generic ingredient
                    before the label scan, so brand names and dosage prefixes
                    (e.g. "Sustained-release Metformin") resolve correctly.

    Returns:
        Dict with keys:
          source          — "OpenFDA drug labels"
          drugs_checked   — the input list (if the check ran)
          interactions    — list of {drugs, source_label, description}
          message         — human-readable status (no interactions / unavailable)
          disclaimer      — always present; states this is a best-effort check
          notes           — any per-drug label-lookup failures
    """
    if not drug_names or len(drug_names) < 2:
        return {
            "source": "OpenFDA drug labels",
            "message": "Need at least 2 drugs to check interactions.",
            "interactions": [],
            "disclaimer": _FDA_INTERACTION_DISCLAIMER,
        }

    interactions: list[dict] = []
    notes: list[str] = []

    try:
        with httpx.Client(timeout=15) as client:
            # Canonicalize each drug name once via RxNorm so brand names and
            # dosage-prefixed strings ('Sustained-release metformin 500mg')
            # resolve to a generic ingredient that the FDA label will mention.
            bases = [_canonical_drug_name(client, d) for d in drug_names]
            indexed = [(i, b) for i, b in enumerate(bases) if b]
            if len(indexed) < 2:
                return {
                    "source": "OpenFDA drug labels",
                    "message": "Could not parse enough drug names to check interactions.",
                    "interactions": [],
                    "disclaimer": _FDA_INTERACTION_DISCLAIMER,
                }

            for i, focal in indexed:
                label = _fetch_label(client, focal)
                if not label:
                    notes.append(f"No FDA label found for '{focal}'.")
                    continue

                text = _interaction_text(label)
                if not text:
                    continue

                for j, other in indexed:
                    if j == i or not other or other == focal:
                        continue
                    idx = text.find(other)
                    if idx < 0:
                        continue
                    start = max(0, idx - 120)
                    end = min(len(text), idx + 220)
                    snippet = text[start:end].strip()
                    interactions.append({
                        "drugs": [drug_names[i], drug_names[j]],
                        "source_label": focal,
                        "description": f"...{snippet}...",
                    })

        # Deduplicate by unordered drug pair — keep first occurrence
        seen: set[tuple[str, str]] = set()
        unique: list[dict] = []
        for it in interactions:
            key = tuple(sorted(d.lower() for d in it["drugs"]))
            if key in seen:
                continue
            seen.add(key)
            unique.append(it)

        if not unique:
            return {
                "source": "OpenFDA drug labels",
                "drugs_checked": list(drug_names),
                "interactions": [],
                "message": "No interactions found in FDA labels (best-effort check).",
                "notes": notes,
                "disclaimer": _FDA_INTERACTION_DISCLAIMER,
            }

        return {
            "source": "OpenFDA drug labels",
            "drugs_checked": list(drug_names),
            "interactions": unique[:5],
            "notes": notes,
            "disclaimer": _FDA_INTERACTION_DISCLAIMER,
        }

    except Exception as e:
        return {
            "source": "OpenFDA drug labels",
            "error": str(e),
            "interactions": [],
            "message": (
                "Drug-interaction lookup is currently unavailable. "
                "Please confirm any interaction questions with your pharmacist."
            ),
            "disclaimer": _FDA_INTERACTION_DISCLAIMER,
        }
