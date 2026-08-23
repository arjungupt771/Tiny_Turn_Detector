import pymupdf
from ai.llm_client import call_qwen, parse_json_response
import json
import policy.manager as manager 



# system_prompt = """
# You are an expert compliance evaluator.

# You will receive:

# 1. REQUIREMENTS_JSON
# 2. REPORT_TEXT

# Your task is to evaluate REPORT_TEXT ONLY against the provided REQUIREMENTS_JSON.

# Rules:
# - Use ONLY the provided requirements.
# - Do NOT use external knowledge.
# - Do NOT assume the existence of policies, processes, or controls that are not explicitly described in REPORT_TEXT.
# - Every score must be supported by evidence found in REPORT_TEXT.
# - If there is no supporting evidence, assign a score of 0 for that requirement.

# Evaluation Process:

# 1. Evaluate every requirement individually.
# 2. Score each requirement semantically:
#    - 100 = Fully addressed
#    - 50 = Partially addressed
#    - 0 = Not addressed
# 3. Calculate:

#    Category Alignment =
#    (Sum of requirement scores / (Number of requirements × 100)) × 100

# 4. For each category:
#    - Include ONLY requirements with a score greater than 0 in "covered".
#    - Combine all supporting evidence into one concise paragraph.
#    - Do NOT repeat similar evidence.
#    - Do NOT list every missing requirement individually.

# 5. Calculate:

#    Overall Alignment =
#    Average of all category alignment percentages.

# 6. Identify the three most important compliance gaps.
#    Prioritize gaps that have the greatest impact on overall compliance.

# Return ONLY valid JSON in the following format:

# {
#   "overall_alignment": 78,
#   "categories": [
#     {
#       "category": "...",
#       "alignment": 80,
#       "covered": [
#         "requirement 1",
#         "requirement 2"
#       ],
#       "evidence": "Concise combined evidence supporting the covered requirements."
#     }
#   ],
#   "top_gaps": [
#     "gap 1",
#     "gap 2",
#     "gap 3"
#   ]
# }

# Do not return explanations, markdown, comments, or additional text.
# Return ONLY valid JSON.
# """


system_prompt = """
You are an expert policy compliance evaluator.

You will receive exactly two inputs:

1. REQUIREMENTS_JSON
2. REPORT_TEXT

Your task is to evaluate REPORT_TEXT strictly against REQUIREMENTS_JSON.

CORE RULES:

- REQUIREMENTS_JSON is the only source of compliance requirements.
- REPORT_TEXT is the only source of evidence.
- Do not use external knowledge.
- Do not introduce requirements, standards, controls, expectations, or gaps that are not contained in REQUIREMENTS_JSON.
- A topic appearing in REPORT_TEXT does NOT become a requirement unless it is explicitly represented in REQUIREMENTS_JSON.
- Do not assume that the absence of a statement proves that a process does not exist. It means there is no supporting evidence in REPORT_TEXT.
- Do not treat general statements as evidence for a more specific requirement.
- Do not require exact wording. Evaluate semantic meaning.
- Do not infer compliance beyond what the evidence supports.

REQUIREMENT EVALUATION:

Evaluate EVERY requirement in REQUIREMENTS_JSON independently.

For each requirement assign exactly one score:

100 = Fully addressed
The REPORT_TEXT contains clear evidence satisfying the requirement.

50 = Partially addressed
The REPORT_TEXT contains relevant evidence, but it satisfies only part of the requirement or lacks an important element.

0 = Not addressed
The REPORT_TEXT contains no sufficient evidence that the requirement is satisfied.

Important:
- A score of 0 means "no sufficient evidence found in REPORT_TEXT".
- Do not claim that the organization does not have a control merely because REPORT_TEXT does not mention it.
- Evidence must support the specific requirement being evaluated.

SECTION / CATEGORY HANDLING:

Treat each "section" in REQUIREMENTS_JSON as a separate category.

For each category:

1. Evaluate all requirements belonging to that section.
2. Calculate:

Category Alignment =
(sum of requirement scores / (number of requirements × 100)) × 100

3. Include the requirement in "covered" only when its score is greater than 0.
4. Summarize the supporting evidence concisely.
5. Do not invent additional requirements from REPORT_TEXT.
6. Do not create a gap merely because REPORT_TEXT discusses a related topic that is not required by REQUIREMENTS_JSON.

OVERALL ALIGNMENT:

Calculate:

Overall Alignment =
average of all category alignment percentages.

Do not use an alignment percentage, score, or conclusion that appears inside REPORT_TEXT.

Do not estimate or subjectively adjust the calculated score.

GAPS:

Identify up to three important gaps.

A gap must correspond to an explicit requirement in REQUIREMENTS_JSON that received a score of 0 or 50.

Prioritize gaps based on:
1. Mandatory requirements with score 0.
2. Mandatory requirements with score 50.
3. Requirements with the greatest compliance impact.

Do not create gaps from topics that exist only in REPORT_TEXT.

EVIDENCE:

For every requirement with a score greater than 0, provide concise evidence from REPORT_TEXT explaining why it is partially or fully addressed.

For requirements scored 0, do not invent evidence.

OUTPUT:

Return ONLY valid JSON.

Use exactly this structure:

{
  "overall_alignment": 78,
  "categories": [
    {
      "category": "Section name",
      "alignment": 80,
      "requirements": [
        {
          "requirement": "Exact requirement description from REQUIREMENTS_JSON",
          "score": 100,
          "status": "Fully addressed",
          "evidence": "Concise evidence from REPORT_TEXT."
        },
        {
          "requirement": "Exact requirement description from REQUIREMENTS_JSON",
          "score": 50,
          "status": "Partially addressed",
          "evidence": "Concise evidence from REPORT_TEXT."
        },
        {
          "requirement": "Exact requirement description from REQUIREMENTS_JSON",
          "score": 0,
          "status": "Not addressed",
          "evidence": ""
        }
      ],
      "covered": [
        "Requirement description 1",
        "Requirement description 2"
      ],
      "evidence": "Concise combined evidence for the covered requirements."
    }
  ],
  "top_gaps": [
    "Explicit requirement from REQUIREMENTS_JSON that is not fully addressed.",
    "Explicit requirement from REQUIREMENTS_JSON that is not fully addressed.",
    "Explicit requirement from REQUIREMENTS_JSON that is not fully addressed."
  ]
}

STRICT OUTPUT REQUIREMENTS:

- Return valid JSON only.
- Do not use Markdown.
- Do not wrap the JSON in ```json or ``` fences.
- Do not add explanations before or after the JSON.
- Preserve the requirement meaning from REQUIREMENTS_JSON.
- Do not omit any requirement from the "requirements" array.
- Do not add requirements that are not present in REQUIREMENTS_JSON.
"""

# multi_framework_final_output = """You are an expert report analysis assistant.

# You will receive a JSON containing the results of a compliance
# assessment for ONE document evaluated against MULTIPLE compliance
# frameworks (policies).

# Analyze the JSON and generate a concise, professional report.

# Your report should include:

# 1. **Overall Summary** - Briefly describe how the document performs
#    across all evaluated frameworks, and which framework it aligns with
#    most closely.
# 2. **Framework Scores** - Present the alignment percentage for every
#    framework evaluated.
# 3. **Strengths** - Highlight the categories and requirements with high
#    compliance, across frameworks.
# 4. **Gaps** - Identify missing or partially addressed requirements,
#    noting which framework(s) they came from.
# 5. **Recommendations** - Provide prioritized, actionable
#    recommendations to improve compliance.
# 6. **Conclusion** - Summarize the organization's current compliance
#    status and the most important next steps.

# Guidelines:

# * Base your analysis only on the provided JSON.
# * Do not invent or assume information.
# * Group similar findings instead of repeating individual requirements.
# * Keep the report concise, objective, and professional.
# * If evidence is available in the JSON, use it to support your observations.

# Return the report in Markdown format with clear section headings.

# Output Requirements:
# - Return ONLY the report.
# - Do NOT include placeholders such as "Prepared by", "Author", "Date",
#   "Contact Information", "Company", "Appendix", "Disclaimer", "Notes",
#   "Feel free to...", "You can customize...", "Adjust as needed."
# - Do NOT add introductions, conclusions outside the requested format,
#   or conversational remarks.
# - Do NOT generate template sections or markdown separators unless
#   explicitly requested.
# - Do NOT use placeholder text such as "[Your Name]", "[Company]",
#   "[Date]", or similar.
# - End the response immediately after the final section of the report.
# - Produce only the requested content with no extra commentary.
# """


def evaluate_against_policy(policy_meta: dict, requirements: dict, report_text: str) -> dict:
    """
    Evaluates a single document against a single policy's requirements.
    Never raises for a "bad" evaluation -- callers get an `error` key
    instead, so one framework failing doesn't take down the rest of a
    multi-policy analysis.
    """
    req_json = json.dumps(requirements)
    header = f"{policy_meta['id']}_REQUIREMENTS_JSON\n"

    evaluation_prompt = f"""
    {system_prompt}


    {header}
    {req_json}
    """

    try:
        evaluation = call_qwen(
            evaluation_prompt,
            f"REQUIREMENTS_JSON:\n{req_json}\n\nREPORT_TEXT:\n{report_text}"
        )
    except Exception as e:
        return {
            "policy_id": policy_meta["id"],
            "display_name": policy_meta["name"],
            "error": str(e),
            "alignment": None,
            "top_gaps": [],
            "categories": [],
            "evaluation": None,
        }

    parsed = parse_json_response(evaluation)
    alignment = parsed.get("overall_alignment") if parsed else None
    top_gaps = (parsed.get("top_gaps") or []) if parsed else []
    categories = (parsed.get("categories") or []) if parsed else []

    return {
        "policy_id": policy_meta["id"],
        "display_name": policy_meta["name"],
        "error": None,
        "evaluation": evaluation,
        "alignment": alignment,
        "top_gaps": top_gaps,
        "categories": categories,
    }


def analyze_single_document(filepath, filename: str = None):
    """
    Runs the full pipeline on a single uploaded file:

      Load active policies -> Evaluate document against EVERY active
      policy -> Determine best-matching framework (automatic framework
      detection) -> Generate a multi-framework report.

    This function does not know or care how many policies exist -- it
    just asks policy_manager for whichever ones are currently enabled.
    """
    display_name = filename or str(filepath)

    doc = pymupdf.open(filepath)
    report_text = "\n".join(page.get_text().strip() for page in doc)
    doc.close()

    if not report_text.strip():
        raise ValueError(f"No extractable text found in '{display_name}'.")

    active_policies = manager.get_active_policies()
    if not active_policies:
        raise ValueError(
            "No compliance policies are enabled. Enable at least one "
            "framework in the Policy Library before running an analysis."
        )

    framework_results = []
    for meta in active_policies:
        try:
            requirements = manager.load_policy_requirements(meta["id"])
        except manager.PolicyError as e:
            framework_results.append({
                "policy_id": meta["id"], "display_name": meta["name"],
                "error": str(e), "alignment": None, "top_gaps": [], "categories": [],
                "evaluation": None,
            })
            continue

        result = evaluate_against_policy(meta, requirements, report_text)
        framework_results.append(result)
        print(f"[{display_name}] {meta['id']}: {result['alignment']}%")

    scored = [r for r in framework_results if isinstance(r["alignment"], (int, float))]
    if not scored:
        raise ValueError(
            f"'{display_name}' could not be evaluated against any enabled framework."
        )

    # Automatic framework detection: the framework this document aligns
    # with most closely.
    best_match = max(scored, key=lambda r: r["alignment"])

    best_alignment = best_match["alignment"]

    report_input = {
        "filename": display_name,
        "best_match_framework": best_match["display_name"],
        "frameworks": [
            {
                "name": r["display_name"],
                "alignment": r["alignment"],
                "categories": r["categories"],
                "top_gaps": r["top_gaps"],
                "error": r["error"],
            }
            for r in framework_results
        ],
    }

   # final_report = call_qwen(multi_framework_final_output, json.dumps(report_input))

    final_report = call_qwen(system_prompt, json.dumps(report_input))
    all_gaps = []
    for r in scored:
        for gap in r["top_gaps"]:
            all_gaps.append(f"[{r['display_name']}] {gap}")

    return {
        "filename": display_name,
        "document": report_text,
        "best_match_policy": {
            "id": best_match["policy_id"],
            "display_name": best_match["display_name"],
        },
        "frameworks": [
            {
                "policy_id": r["policy_id"],
                "display_name": r["display_name"],
                "alignment": r["alignment"],
                "categories": r["categories"],
                "top_gaps": r["top_gaps"],
                "error": r["error"],
            }
            for r in framework_results
        ],
        "evaluation": best_match["evaluation"],
        "report": final_report,
        "alignment": best_alignment,
        "top_gaps": all_gaps[:10],
        "categories": best_match["categories"],
    }
