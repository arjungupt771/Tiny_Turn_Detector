from services.compliance_service import analyze_single_document
from services.report_service import generate_combined_summary
from services.chat_service import create_chat_session

def analyze_documents(filepaths):
    """
    filepaths: list of (filename, path) tuples.
    Analyzes each document independently against every enabled policy,
    then synthesizes a combined organizational executive summary.
    Individual document failures are collected and skipped rather than
    aborting the whole batch.
    """
    results = []
    errors = []

    for filename, path in filepaths:
        try:
            result = analyze_single_document(path, filename=filename)
            results.append(result)
        except Exception as e:
            print(f"Failed to analyze '{filename}': {e}")
            errors.append((filename, str(e)))

    if not results:
        raise ValueError("None of the uploaded documents could be analyzed.")

    overall_summary, overall_alignment = generate_combined_summary(results, errors)

    documents = {r["filename"]: r["document"] for r in results}
    reports = {r["filename"]: r["report"] for r in results}
    evaluations = {r["filename"]: r["evaluation"] for r in results}
    policies = {r["filename"]: r["best_match_policy"] for r in results}
    alignment_scores = {r["filename"]: r["alignment"] for r in results}
    frameworks = {r["filename"]: r["frameworks"] for r in results}
    top_gaps = {r["filename"]: r["top_gaps"] for r in results}
    categories = {r["filename"]: r["categories"] for r in results}

    session_id = create_chat_session(
        documents=documents,
        reports=reports,
        evaluations=evaluations,
        policies=policies,
        alignment_scores=alignment_scores,
        overall_summary=overall_summary,
        frameworks=frameworks,
        top_gaps=top_gaps,
        categories=categories,
    )

    return {
        "session_id": session_id,
        "overall_summary": overall_summary,
        "overall_alignment": overall_alignment,
        "documents": [
            {
                "filename": r["filename"],
                "policy": r["best_match_policy"]["display_name"],
                "alignment": r["alignment"],
                "report": r["report"],
                "frameworks": r["frameworks"],
            }
            for r in results
        ],
        "errors": [{"filename": name, "message": msg} for name, msg in errors],
    }
