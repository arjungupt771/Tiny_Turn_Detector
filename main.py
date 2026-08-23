import uvicorn
from pathlib import Path
import shutil
import os
import uuid
import tempfile
from gbvh_test import analyze_documents, generate_pdf, get_chat_session, cleanup_sessions, CHAT_SESSIONS
from services.chat_service import get_or_create_session_id, delete_temp_file
from ai.chat import answer_question
import policy.manager as manager
import ai.copilot as copilot
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi import Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from fastapi.staticfiles import StaticFiles
from storage.collections import create_collections
from urllib.parse import quote, unquote


SESSION_TTL = 86400


app = FastAPI()



# Session middleware
@app.middleware("http")
async def session_middleware(request: Request, call_next):
    session_id, is_new_session = get_or_create_session_id(request)

    request.state.session_id = session_id

    response = await call_next(request)

    if is_new_session:
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            secure=False,   # set True in HTTPS production
            samesite="Lax",
            max_age=SESSION_TTL
        )

    return response


create_collections()

app.mount("/static", StaticFiles(directory="static"), name="static")
BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = Path("/tmp/upload_report")

TMP_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx"}


@app.get("/", response_class=HTMLResponse)
async def home():
    html = (BASE_DIR / "test.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.on_event("startup")
async def on_startup():
    create_collections()

@app.post("/upload")
async def upload(request: Request,files: list[UploadFile] = File(...)):
    # Opportunistic cleanup of old in-memory sessions on new activity.
    # cleanup_sessions()

    session_id = request.state.session_id

    # print("x-x"*30)
    # print(session_id)
    # print("x-x"*30)

    if not files:
        return {"success": False, "message": "No files were provided."}

    # Each upload batch gets its own folder so concurrent users / repeated
    # uploads never collide or overwrite each other's files.
    # upload_id = str(uuid.uuid4())
    # upload_dir = TMP_DIR / upload_id
    upload_dir = TMP_DIR 
    upload_dir.mkdir(parents=True, exist_ok=True)

    
    saved_files = []
    rejected_files = []

    for file in files:
        original_name = os.path.basename(file.filename)
        
        name, ext = os.path.splitext(original_name)
    
        ext = ext.lower()
    
        # final stored filename
        file_name = f"{session_id}<>{original_name}"
    
        file_path = os.path.join(
            upload_dir,
            file_name
        )
    
        # ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            rejected_files.append(file.filename)
            continue

        # dest_path = upload_dir / file.filename
        try:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_files.append(file.filename)
        finally:
            await file.close()

    if not saved_files:
        return {
            "success": False,
            "message": "No supported files were uploaded (allowed: .pdf, .doc, .docx).",
            "rejected_files": rejected_files,
        }

    # print("xx"*30)
    # print(f"inside updated uploaded_file_name: {file_name}")
    # print("xx"*30)

    response = JSONResponse({
        "success": True,
        "filenames": saved_files,
        "rejected_files": rejected_files,})

    response.set_cookie(
            key="uploaded_file_name",
            value=quote(file_name, safe=""),
            httponly=True,
            samesite="Lax",
            secure=False
        )

    return response
    # return {
    #     "success": True,
    #     "filenames": saved_files,
    #     "rejected_files": rejected_files,
    # }

@app.get("/policy-library", response_class=HTMLResponse)
async def policy_library_page():
    html = (BASE_DIR / "policy_library.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# Phase 2 - Policy Library API
# ---------------------------------------------------------------------------

ALLOWED_POLICY_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@app.get("/policies")
async def list_policies():
    """Every policy package on disk, enabled or not."""
    policies =  manager.list_policies(include_disabled=True)
    if policies == []:
        return {"success": False, "policies":"no policies uploaded"}
    
    return {"success": True, "policies": manager.list_policies(include_disabled=True)}


@app.get("/policies/{policy_id}")
async def get_policy_detail(policy_id: str):
    try:
        return {"success": True, "policy": manager.get_policy(policy_id)}
    except manager.PolicyError as e:
        return {"success": False, "message": str(e)}


class PolicyEnableRequest(BaseModel):
    enabled: bool


@app.post("/policies/{policy_id}/enabled")
async def set_policy_enabled(policy_id: str, req: PolicyEnableRequest):
    try:
        metadata = manager.set_policy_enabled(policy_id, req.enabled)
        return {"success": True, "policy": metadata}
    except manager.PolicyError as e:
        return {"success": False, "message": str(e)}


@app.delete("/policies/{policy_id}")
async def delete_policy(policy_id: str):
    try:
        manager.delete_policy(policy_id)

        return {"success": True}
    except manager.PolicyError as e:
        return {"success": False, "message": str(e)}


@app.post("/policies/upload")
async def upload_policy(
    file: UploadFile = File(...),
    name: str = Form(...),
    category: str = Form("Custom"),
    country: str = Form("Global"),
    description: str = Form(""),
    version: str = Form(""),
):
    """
    Admin -> Upload PDF/DOCX/TXT -> Extract text -> Generate metadata ->
    Store policy -> Ready for analysis. No code changes, no deployment.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_POLICY_EXTENSIONS:
        return {
            "success": False,
            "message": f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_POLICY_EXTENSIONS)}",
        }

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        await file.close()

        metadata = manager.create_policy_from_upload(
            file_path=tmp_path,
            original_filename=file.filename,
            name=name,
            category=category,
            country=country,
            description=description,
            version=version,
        )
        return {"success": True, "policy": metadata}
    except manager.PolicyError as e:
        return {"success": False, "message": str(e)}
    except Exception as e:
        return {"success": False, "message": f"Failed to process policy: {e}"}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Phase 3 - AI Compliance Copilot API
# ---------------------------------------------------------------------------

class CopilotRequest(BaseModel):
    session_id: str
    filename: Optional[str] = None


class SimulatorRequest(BaseModel):
    session_id: str
    proposed_change: str
    filename: Optional[str] = None


class RewriteRequest(BaseModel):
    section_text: str


def _get_session_or_error(session_id: str):
    session = get_chat_session(session_id)
    if session is None:
        return None, {"success": False, "message": "Invalid session. Please start a new analysis."}
    return session, None


@app.post("/copilot/summary")
async def copilot_summary(req: CopilotRequest):
    session, error = _get_session_or_error(req.session_id)
    if error:
        return error
    return {"success": True, **copilot.get_copilot_summary(session, req.filename)}


@app.post("/copilot/generate-clauses")
async def copilot_generate_clauses(req: CopilotRequest):
    session, error = _get_session_or_error(req.session_id)
    if error:
        return error
    try:
        return {"success": True, **copilot.generate_missing_clauses(session, req.filename)}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/copilot/explain-gaps")
async def copilot_explain_gaps(req: CopilotRequest):
    session, error = _get_session_or_error(req.session_id)
    if error:
        return error
    try:
        return {"success": True, **copilot.explain_gaps(session, req.filename)}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/copilot/roadmap")
async def copilot_roadmap(req: CopilotRequest):
    session, error = _get_session_or_error(req.session_id)
    if error:
        return error
    try:
        return {"success": True, **copilot.build_roadmap(session, req.filename)}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/copilot/explain-score")
async def copilot_explain_score(req: CopilotRequest):
    session, error = _get_session_or_error(req.session_id)
    if error:
        return error
    return {"success": True, **copilot.explain_score(session, req.filename)}


@app.post("/copilot/suggested-questions")
async def copilot_suggested_questions(req: CopilotRequest):
    session, error = _get_session_or_error(req.session_id)
    if error:
        return error
    return {"success": True, "questions": copilot.suggested_questions(session, req.filename)}


@app.post("/copilot/rewrite")
async def copilot_rewrite(req: RewriteRequest):
    try:
        return {"success": True, **copilot.ai_rewrite_section(req.section_text)}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/copilot/simulate")
async def copilot_simulate(req: SimulatorRequest):
    session, error = _get_session_or_error(req.session_id)
    if error:
        return error
    try:
        return {"success": True, **copilot.compliance_simulator(session, req.proposed_change, req.filename)}
    except Exception as e:
        return {"success": False, "message": str(e)}


class GenerateRequest(BaseModel):
    upload_id: str

class PdfRequest(BaseModel):
    pdf_path: str
    download_name: str = "GBVH_Compliance_Report.pdf"

class ReportRequest(BaseModel):
    session_id: str

class ChatRequest(BaseModel):
    message: str

class NewSessionRequest(BaseModel):
    session_id: str

@app.post("/chat")
async def chat(request: Request,req: ChatRequest):
    session_id = request.state.session_id
    # file_name = request.cookies.get("uploaded_file_name")
    file_name = unquote(request.cookies.get("uploaded_file_name", ""))
    result = answer_question(
        session_id,
        req.message,
        file_name
    )
    return result

@app.get("/report", response_class=HTMLResponse)
async def report_page():
    html = (BASE_DIR / "report.html").read_text(
        encoding="utf-8"
    )
    return HTMLResponse(content=html)

@app.post("/get-report")
async def get_report(req: ReportRequest):

    session = get_chat_session(req.session_id)

    if session is None:

        return {
            "success": False,
            "message": "Invalid session."
        }

    return {

        "success": True,

        "report": session["overall_summary"]

    }

# @app.post("/download-pdf")
# async def download_pdf(request: PdfRequest):
#     return FileResponse(
#         path=request.pdf_path,
#         media_type="application/pdf",
#         filename=request.download_name
#     )

@app.get("/download-pdf")
async def download_pdf(
    pdf_path: str = Query(...),
    download_name: str = Query(...)):
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=download_name
    )


@app.delete("/remove-uploaded-file")
async def remove_uploaded_file(request: Request):
    session_id = request.state.session_id

    uploaded_file_name = unquote(request.cookies.get("uploaded_file_name", ""))
    # request.cookies.get("uploaded_file_name")

    # print("xx"*30)
    # print(f"uploaded_file_name: {uploaded_file_name}")
    # print("xx"*30)

    if uploaded_file_name:
        expected_prefix = f"{session_id}<>"

        # print("xx"*30)
        # print(f"expected_prefix: {expected_prefix}")
        # print("xx"*30)

        if uploaded_file_name.startswith(expected_prefix):
            file_path = os.path.join(
                TMP_DIR,
                uploaded_file_name
            )

            # print("xx"*30)
            # print(f"uploaded_file_name: {uploaded_file_name}")
            # print("xx"*30)
            
            # delete_temp_file(file_path)

    response = JSONResponse({
        "message": "Uploaded file removed"
    })

    response.delete_cookie(
        key="uploaded_file_name",
        path="/"
    )

    return response



@app.post("/new-session")
async def new_session(request: NewSessionRequest):
    """
    Clear the current session and all associated data.
    Deletes chat history, PDF files, and session metadata.
    """
    session_id = request.session_id
    
    # Get session to find the PDF path before deleting
    session = CHAT_SESSIONS.get(session_id)
    
    # Delete from in-memory session store
    if session_id in CHAT_SESSIONS:
        del CHAT_SESSIONS[session_id]
    
    return {
        "success": True,
        "message": "Session cleared successfully"
    }

@app.post("/generate")
async def generate(req: GenerateRequest):
    upload_dir = TMP_DIR / req.upload_id

    if not upload_dir.exists() or not upload_dir.is_dir():
        return {
            "success": False,
            "message": "Upload batch not found. Please re-upload your document(s)."
        }

    filepaths = [
        (p.name, p) for p in sorted(upload_dir.iterdir()) if p.is_file()
    ]

    if not filepaths:
        return {
            "success": False,
            "message": "No files found for this upload."
        }

    try:
        result = analyze_documents(filepaths)

        documents_out = []
        for d in result["documents"]:
            pdf_path = generate_pdf(d["report"], file_name=f"{uuid.uuid4()}.pdf")
            documents_out.append({
                "filename": d["filename"],
                "policy": d["policy"],
                "alignment": d["alignment"],
                "report": d["report"],
                "frameworks": d.get("frameworks", []),
                "pdf_path": pdf_path,
            })

        overall_pdf_path = generate_pdf(result["overall_summary"], file_name=f"{uuid.uuid4()}.pdf") 

        return {
            "success": True,
            "session_id": result["session_id"],
            "overall_summary": result["overall_summary"],
            "overall_alignment": result["overall_alignment"],
            "overall_pdf_path": overall_pdf_path,
            "documents": documents_out,
            "errors": result["errors"],
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }
    finally:
        # The files have been read into memory for analysis; clean up disk.
        shutil.rmtree(upload_dir, ignore_errors=True)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",      # replace "main" with your filename if different
        host="0.0.0.0",
        port=8012,
        reload=True,
    )


