import json

from ai.llm_client import call_qwen, parse_json_response


def _doc_context(session, filename: str = None):
    """Returns (label, report_text, top_gaps, categories, frameworks, alignment)."""
    if filename and filename in session["documents"]:
        return (
            filename,
            session["reports"].get(filename, ""),
            session["top_gaps"].get(filename, []),
            session["categories"].get(filename, []),
            session["frameworks"].get(filename, []),
            session["alignment_scores"].get(filename),
        )
    # organization-wide fallback
    all_gaps = []
    for gaps in session["top_gaps"].values():
        all_gaps.extend(gaps)
    return (
        "Organization (all documents)",
        session["overall_summary"],
        all_gaps,
        [],
        [],
        None,
    )


# ---------------------------------------------------------------------------
# Phase 3.1 - Copilot card: critical issues + suggested actions
# ---------------------------------------------------------------------------

def get_copilot_summary(session, filename: str = None) -> dict:
    label, _, top_gaps, _, frameworks, alignment = _doc_context(session, filename)
    return {
        "scope": label,
        "alignment": alignment,
        "critical_issues": top_gaps[:5],
        "frameworks": frameworks,
        "suggested_actions": [
            {"id": "generate_clauses", "label": "Generate Clauses"},
            {"id": "explain_gaps", "label": "Explain Gaps"},
            {"id": "build_roadmap", "label": "Build Roadmap"},
        ],
    }


# ---------------------------------------------------------------------------
# Phase 3.2 - Generate Missing Clauses
# ---------------------------------------------------------------------------

clause_prompt = """You are a policy drafting assistant for workplace
compliance documents.

You will receive a list of compliance gaps identified in an
organization's document. For each gap, draft a short, ready-to-insert
policy clause that would address it.

Return ONLY valid JSON in this shape:

[
  {"title": "Anonymous Reporting", "clause": "Employees shall have access to a confidential, anonymous reporting channel for..."}
]

Guidelines:
- One clause per gap, in the same order as the gaps given.
- Each clause should be 2-5 sentences, written in formal policy
  language ("Employees shall...", "The organization will...").
- Do not include commentary outside the JSON array.
"""


def generate_missing_clauses(session, filename: str = None) -> dict:
    label, _, top_gaps, _, _, _ = _doc_context(session, filename)
    if not top_gaps:
        return {"scope": label, "clauses": []}

    raw = call_qwen(clause_prompt, json.dumps(top_gaps[:8]))
    parsed = parse_json_response(raw)
    clauses = parsed if isinstance(parsed, list) else []
    return {"scope": label, "clauses": clauses}


# ---------------------------------------------------------------------------
# Phase 3.3 - Explain Gaps
# ---------------------------------------------------------------------------

explain_gaps_prompt = """You are a compliance consultant explaining
gaps to a non-expert audience.

You will receive a list of compliance gaps, each possibly tagged with
the framework it came from (in square brackets, e.g.
"[UN GBVH Framework] Missing Investigation Timeline").

For each gap, explain:
- why_it_matters: one or two sentences.
- business_risk: one sentence on the concrete risk of leaving it unaddressed.
- referenced_framework: the framework name if given, otherwise "General best practice".

Return ONLY valid JSON in this shape:

[
  {
    "gap": "Missing Investigation Timeline",
    "why_it_matters": "...",
    "business_risk": "...",
    "referenced_framework": "..."
  }
]
"""


def explain_gaps(session, filename: str = None) -> dict:
    label, _, top_gaps, _, _, _ = _doc_context(session, filename)
    if not top_gaps:
        return {"scope": label, "gaps": []}

    raw = call_qwen(explain_gaps_prompt, json.dumps(top_gaps[:8]))
    parsed = parse_json_response(raw)
    gaps = parsed if isinstance(parsed, list) else []
    return {"scope": label, "gaps": gaps}


# ---------------------------------------------------------------------------
# Phase 3.4 - Improvement Roadmap
# ---------------------------------------------------------------------------

roadmap_prompt = """You are a compliance program manager.

You will receive a JSON object with the current compliance alignment
percentage and a list of gaps.

Build a 30-day improvement plan broken into 4 weekly milestones. Each
week should have a short title and 2-4 concrete tasks that address the
given gaps, in priority order (most critical / highest-risk first).

Also estimate a realistic predicted compliance percentage after the
30 days if the plan is followed (a modest, defensible increase, not a
guarantee).

Return ONLY valid JSON in this shape:

{
  "weeks": [
    {"week": 1, "title": "...", "tasks": ["...", "..."]},
    {"week": 2, "title": "...", "tasks": ["...", "..."]},
    {"week": 3, "title": "...", "tasks": ["...", "..."]},
    {"week": 4, "title": "...", "tasks": ["...", "..."]}
  ],
  "predicted_alignment": 76
}
"""


def build_roadmap(session, filename: str = None) -> dict:
    label, _, top_gaps, _, _, alignment = _doc_context(session, filename)

    payload = {"current_alignment": alignment, "gaps": top_gaps[:10]}
    raw = call_qwen(roadmap_prompt, json.dumps(payload))
    parsed = parse_json_response(raw) or {}

    return {
        "scope": label,
        "current_alignment": alignment,
        "predicted_alignment": parsed.get("predicted_alignment"),
        "weeks": parsed.get("weeks", []),
    }


# ---------------------------------------------------------------------------
# Phase 3.5 - Explain Compliance Score
# ---------------------------------------------------------------------------

def explain_score(session, filename: str = None) -> dict:
    """
    Deterministic, not an LLM call: buckets each evaluated category by
    its alignment percentage so the score breakdown is always
    trustworthy and instant. Present >= 80, Partial 30-79, Missing < 30.
    """
    label, _, _, categories, frameworks, alignment = _doc_context(session, filename)

    def bucket(pct):
        if pct is None:
            return "Unknown"
        if pct >= 80:
            return "Present"
        if pct >= 30:
            return "Partial"
        return "Missing"

    breakdown = [
        {
            "category": c.get("category"),
            "alignment": c.get("alignment"),
            "status": bucket(c.get("alignment")),
        }
        for c in categories
    ]

    return {
        "scope": label,
        "overall_alignment": alignment,
        "frameworks": frameworks,
        "breakdown": breakdown,
    }


# ---------------------------------------------------------------------------
# Phase 3.6 - Smart Suggested Questions
# ---------------------------------------------------------------------------

def suggested_questions(session, filename: str = None) -> list:
    """
    Deterministic and instant (no LLM call) -- these are meant to
    populate the chat input as clickable chips the moment the report
    is ready.
    """
    label, _, top_gaps, _, frameworks, _ = _doc_context(session, filename)

    questions = ["Why is my score low?", "Generate missing clauses."]

    if top_gaps:
        questions.append(f"Explain the gap: {top_gaps[0]}")
    if len(frameworks) > 1:
        questions.append("Compare my scores across frameworks.")
    questions.append("Build a 30-day improvement roadmap.")

    return questions[:5]


# ---------------------------------------------------------------------------
# Phase 3.7 - AI Rewrite
# ---------------------------------------------------------------------------

rewrite_prompt = """You are a policy editor improving a section of a
workplace compliance document so it better satisfies compliance
frameworks such as UN GBVH and ILO C190.

You will receive the original section text.

Return ONLY valid JSON in this shape:

{
  "improved": "the rewritten section text",
  "reason": "one or two sentences on what changed and why it's more compliant"
}

Keep the improved version close in length and tone to the original --
this is an edit, not a rewrite from scratch.
"""


def ai_rewrite_section(section_text: str) -> dict:
    if not section_text or not section_text.strip():
        raise ValueError("No section text provided to rewrite.")

    raw = call_qwen(rewrite_prompt, section_text)
    parsed = parse_json_response(raw) or {}

    return {
        "original": section_text,
        "improved": parsed.get("improved", ""),
        "reason": parsed.get("reason", ""),
    }


# ---------------------------------------------------------------------------
# Phase 3.8 - Compliance Simulator
# ---------------------------------------------------------------------------

simulator_prompt = """You are a compliance analyst running a
"what-if" simulation.

You will receive the current compliance alignment percentage, the
current list of gaps, and a proposed change the organization is
considering.

Estimate a realistic predicted alignment percentage if the proposed
change were implemented, and explain briefly why, referencing which
gaps it would close.

Return ONLY valid JSON in this shape:

{
  "predicted_alignment": 52,
  "reason": "Improves compliance with reporting and accessibility requirements under the selected frameworks."
}

Be conservative and realistic -- a single change rarely closes every
gap.
"""


def compliance_simulator(session, proposed_change: str, filename: str = None) -> dict:
    label, _, top_gaps, _, _, alignment = _doc_context(session, filename)

    if not proposed_change or not proposed_change.strip():
        raise ValueError("No proposed change provided.")

    payload = {
        "current_alignment": alignment,
        "gaps": top_gaps[:10],
        "proposed_change": proposed_change,
    }
    raw = call_qwen(simulator_prompt, json.dumps(payload))
    parsed = parse_json_response(raw) or {}

    return {
        "scope": label,
        "current_alignment": alignment,
        "predicted_alignment": parsed.get("predicted_alignment"),
        "reason": parsed.get("reason", ""),
    }
