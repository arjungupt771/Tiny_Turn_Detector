
import time
import uuid
import secrets
from fastapi import Request
import os

CHAT_SESSIONS = {}


def get_or_create_session_id(request: Request) -> tuple[str, bool]:
    """
    Returns:
    - session_id
    - is_new_session
    """
    session_id = request.cookies.get("session_id")

    if session_id:
        return session_id, False

    return str(secrets.token_urlsafe(32)), True


def delete_temp_file(file_path: str):
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"[CLEANUP] Deleted temp file: {file_path}")
    except Exception as e:
        print(f"[CLEANUP] Failed to delete temp file: {e}")




def create_chat_session(documents, reports, evaluations, policies, alignment_scores,
                         overall_summary, frameworks=None, top_gaps=None, categories=None):
    """
    CHAT_SESSIONS[session_id] = {
        "documents": {filename: full_extracted_text, ...},
        "reports": {filename: per_document_markdown_report, ...},
        "evaluations": {filename: raw_evaluation_json_text_of_best_match, ...},
        "policies": {filename: {"id":..., "display_name": ...}, ...}  (best match),
        "alignment_scores": {filename: float | None, ...},
        "frameworks": {filename: [{"policy_id","display_name","alignment","categories","top_gaps"}...]},
        "top_gaps": {filename: [str, ...]},
        "categories": {filename: [...]},
        "overall_summary": str,
        "history": [...],
        "created_at": float,
        "status": "completed",
    }
    """
    session_id = str(uuid.uuid4())

    CHAT_SESSIONS[session_id] = {
        "documents": documents,
        "reports": reports,
        "evaluations": evaluations,
        "policies": policies,
        "alignment_scores": alignment_scores,
        "frameworks": frameworks or {},
        "top_gaps": top_gaps or {},
        "categories": categories or {},
        "overall_summary": overall_summary,
        "history": [],
        "created_at": time.time(),
        "status": "completed"
    }
    return session_id


def get_chat_session(session_id):
    return CHAT_SESSIONS.get(session_id)


def cleanup_sessions():
    now = time.time()
    expired = []
    for sid, session in CHAT_SESSIONS.items():
        if now - session["created_at"] > 1800:
            expired.append(sid)
    for sid in expired:
        del CHAT_SESSIONS[sid]


def update_chat_history(session_id, role, content):
    CHAT_SESSIONS[session_id]["history"].append({
        "role": role,
        "content": content
    })