from ai.llm_client import call_qwen
import json



combined_summary_prompt = """You are an expert organizational compliance analyst.

You will receive a JSON array. Each item represents one document that has
already been evaluated against one or more compliance frameworks, and
contains:
- filename
- frameworks (list of {name, alignment})
- best_match_framework
- overall_alignment (a percentage already calculated - do not recompute it)
- top_gaps

You will also receive a pre-calculated "organization_overall_alignment"
percentage that averages all documents' best-match alignment - use this
exact number, do not recalculate it.

Using ONLY the provided JSON, write ONE executive summary for the
organization as a whole, in Markdown, with exactly these sections:

## Strengths
Cross-document patterns where compliance is strong. Group similar points
instead of repeating per-document detail.

## Major Gaps
Cross-document patterns of missing or weak compliance. Prioritize gaps that
appear in multiple documents, multiple frameworks, or are high-severity.

## Priority Recommendations
A numbered list of 3-5 concrete, actionable, prioritized recommendations for
the organization, ordered by importance.

Guidelines:
- Base your analysis only on the provided JSON. Do not invent findings.
- Do not repeat the organization_overall_alignment number back verbatim as
  its own section - it will be shown separately.
- Do not add introductions, disclaimers, or conversational remarks.
- Do not use placeholder text.
- Return ONLY the three sections above in Markdown.
"""


def generate_combined_summary(results, errors=None):
    """
    Produces the organization-wide executive summary across all
    uploaded documents and all frameworks they were evaluated against.
    """
    errors = errors or []

    alignments = [r["alignment"] for r in results if isinstance(r["alignment"], (int, float))]
    overall_alignment = round(sum(alignments) / len(alignments), 1) if alignments else None

    synthesis_input = {
        "organization_overall_alignment": overall_alignment,
        "documents": [
            {
                "filename": r["filename"],
                "best_match_framework": r["best_match_policy"]["display_name"],
                "frameworks": [
                    {"name": f["display_name"], "alignment": f["alignment"]}
                    for f in r["frameworks"]
                ],
                "overall_alignment": r["alignment"],
                "top_gaps": r["top_gaps"],
            }
            for r in results
        ],
    }

    try:
        narrative = call_qwen(
            combined_summary_prompt,
            json.dumps(synthesis_input),
        )
    except Exception as e:
        print(f"Combined summary synthesis failed, falling back to a basic summary: {e}")
        narrative_lines = ["## Major Gaps"]
        for r in results:
            for gap in r["top_gaps"][:3]:
                narrative_lines.append(f"- ({r['filename']}) {gap}")
        narrative = "\n".join(narrative_lines) if len(narrative_lines) > 1 else "## Summary\nNo synthesized summary available."

    header_lines = ["# Overall Organizational Compliance Summary", ""]
    header_lines.append(
        f"**Overall Organizational Alignment: {overall_alignment if overall_alignment is not None else 'N/A'}%**"
    )
    header_lines.append("")
    header_lines.append("## Document Alignment Scores")
    for r in results:
        alignment = r["alignment"]
        fw_scores = ", ".join(
            f"{f['display_name']} {f['alignment']}%"
            for f in r["frameworks"] if isinstance(f["alignment"], (int, float))
        )
        header_lines.append(
            f"- **{r['filename']}** (best match: {r['best_match_policy']['display_name']}): "
            f"{alignment if alignment is not None else 'N/A'}% overall  \n"
            f"  _All frameworks: {fw_scores}_"
        )
    header_lines.append("")

    if errors:
        header_lines.append("## Skipped Documents")
        for name, reason in errors:
            header_lines.append(f"- **{name}**: {reason}")
        header_lines.append("")

    summary_markdown = "\n".join(header_lines) + "\n" + narrative

    return summary_markdown, overall_alignment
