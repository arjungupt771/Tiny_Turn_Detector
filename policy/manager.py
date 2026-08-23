import json
import re,os
import shutil
import pymupdf
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Import Docling's native structural types
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import TextItem, TableItem, PictureItem, DocItemLabel
from docling.exceptions import ConversionError
import time
import re

from qdrant_client.models import Filter, FieldCondition, MatchValue
from storage.qdrant_client import get_qdrant_client


from ai.llm_client import call_qwen, parse_json_response
from storage.uploader import save_policy as qdrant_save_requirements, delete_policy as qdrant_delete_requirements, policy_info, policy_aliginment

from storage.config import POLICY_INFO


BASE_DIR = Path(__file__).resolve().parent.parent
POLICIES_DIR =  BASE_DIR/ "policies"
POLICIES_DIR.mkdir(exist_ok=True)

METADATA_FILE = "metadata.json"
REQUIREMENTS_FILE = "requirements.json"

REQUIRED_METADATA_FIELDS = {
    "id", "name", "version", "category", "description",
    "country", "enabled", "uploaded_at", "total_sections",
}


pipeline_options = PdfPipelineOptions()
pipeline_options.generate_page_images = True      # <-- Correct
pipeline_options.generate_picture_images = True
pipeline_options.images_scale = 2.0

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)


# extraction_prompt = """
# You are a compliance-policy structuring assistant.

# You will receive the raw text of a policy or regulatory framework
# document. Extract its substantive requirements and group them into
# logical categories, in the same style as an existing compliance
# checklist.

# Rules:
# - Identify 5-15 categories that cover the document's requirements
#   (e.g. "reporting_and_response", "training_and_awareness",
#   "monitoring_and_accountability").
# - Category keys must be lowercase, use underscores instead of spaces,
#   and contain no punctuation.
# - Under each category, list the specific requirements as short phrases
#   (3-8 words each), not full sentences.
# - Base this only on the provided document text. Do not invent
#   requirements the document does not support.
# - Do not include commentary, explanations, or anything other than the
#   JSON object.

# Return ONLY valid JSON in this exact shape:

# {
#   "category_one": ["requirement 1", "requirement 2"],
#   "category_two": ["requirement 1", "requirement 2"]
# }
# """


extraction_prompt = """
You are an expert compliance policy analyst.

You will receive the full text of a compliance policy, regulation, framework, or standard.

Your task is to extract EVERY explicit compliance obligation contained in the document.

Rules:

1. Extract only requirements that are explicitly supported by the document.
2. Every extracted requirement must be a complete standalone sentence.
3. Preserve the original meaning of the policy.
4. Do NOT reduce requirements to keywords.
5. Split compound requirements into multiple requirements whenever they describe different obligations.
6. Group requirements into logical compliance categories.
7. Category names must be lowercase with underscores.
8. Do NOT invent requirements.
9. Return ONLY valid JSON.

Example:

Policy:
"The employer shall establish confidential reporting channels and protect complainants from retaliation."

Output:

{
  "reporting_and_response": [
    "The employer shall establish confidential reporting channels.",
    "The employer shall protect complainants from retaliation."
  ]
}
"""


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or f"policy_{uuid.uuid4().hex[:8]}"


def _unique_policy_id(base_slug: str) -> str:
    candidate = base_slug
    suffix = 2
    existing = {p.name for p in POLICIES_DIR.iterdir() if p.is_dir()}
    while candidate in existing:
        candidate = f"{base_slug}_{suffix}"
        suffix += 1
    return candidate


def _policy_dir(policy_id: str) -> Path:
    return POLICIES_DIR / policy_id


def _total_sections(requirements: dict) -> int:
    return sum(len(v) for v in requirements.values() if isinstance(v, list))


class PolicyError(Exception):
    pass


def validate_policy_package(policy_id: str) -> None:
    """
    Raises PolicyError if a policy folder is missing required files or
    the metadata is malformed. Used on load so one bad/hand-edited
    folder never crashes the whole platform.
    """
    pdir = _policy_dir(policy_id)
    meta_path = pdir / METADATA_FILE
    req_path = pdir / REQUIREMENTS_FILE

    if not meta_path.exists():
        raise PolicyError(f"Policy '{policy_id}' is missing {METADATA_FILE}.")
    if not req_path.exists():
        raise PolicyError(f"Policy '{policy_id}' is missing {REQUIREMENTS_FILE}.")

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise PolicyError(f"Policy '{policy_id}' has invalid metadata.json: {e}")

    missing = REQUIRED_METADATA_FIELDS - set(metadata.keys())
    if missing:
        raise PolicyError(f"Policy '{policy_id}' metadata.json missing fields: {sorted(missing)}")

    try:
        requirements = json.loads(req_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise PolicyError(f"Policy '{policy_id}' has invalid requirements.json: {e}")

    if not isinstance(requirements, dict) or not requirements:
        raise PolicyError(f"Policy '{policy_id}' requirements.json must be a non-empty object.")


def load_policy_metadata(policy_id: str) -> dict:
    validate_policy_package(policy_id)
    meta_path = _policy_dir(policy_id) / METADATA_FILE
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_policy_requirements(policy_id: str) -> dict:
    validate_policy_package(policy_id)
    req_path = _policy_dir(policy_id) / REQUIREMENTS_FILE
    return json.loads(req_path.read_text(encoding="utf-8"))


def get_policy(policy_id: str) -> dict:
    return {
        "metadata": load_policy_metadata(policy_id),
        "requirements": load_policy_requirements(policy_id),
    }


# def list_policies(include_disabled: bool = True) -> list[dict]:
#     """
#     Returns metadata for every policy package on disk. Bad/corrupt
#     packages are skipped (with a warning) rather than breaking the
#     whole listing.
#     """
#     results = []
#     if not POLICIES_DIR.exists():
#         return results

#     for pdir in sorted(POLICIES_DIR.iterdir()):
#         if not pdir.is_dir():
#             continue
#         try:
#             metadata = load_policy_metadata(pdir.name)
#         except PolicyError as e:
#             print(f"[policy_manager] Skipping invalid policy '{pdir.name}': {e}")
#             continue

#         if not include_disabled and not metadata.get("enabled", True):
#             continue

#         results.append(metadata)

#     return results


def list_policies(include_disabled: bool = True) -> list[dict]:
    """
    Returns policy metadata from the policy_info collection.

    Example:
    [
        {
            "policy_id": "un_policy",
            "region": "india",
            "version": "1",
            "status": "selected"
        },
        ...
    ]
    """
    client = get_qdrant_client()

    if not client.collection_exists(POLICY_INFO):
        return []

    policies = []
    next_page = None

    scroll_filter = None
    if not include_disabled:
        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key="enabled",
                    match=MatchValue(value=True)
                )
            ]
        )

    while True:
        points, next_page = client.scroll(
            collection_name=POLICY_INFO,
            scroll_filter=scroll_filter,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=next_page,
        )

        for point in points:
            payload = point.payload


            policies.append({
                "id": payload.get("policy_id"),
                "name": payload.get("policy_id"),
                "country": payload.get("region"),
                "version": payload.get("version"),
                "enabled": payload.get("status") == "selected",
            })

        if next_page is None:
            break

    return policies


def get_active_policies() -> list[dict]:
    """Metadata for every enabled policy. This is what the compliance
    analyzer uses -- it never needs to know how many policies exist
    or hardcode any framework name."""
    return list_policies(include_disabled=False)


# def set_policy_enabled(policy_id: str, enabled: bool) -> dict:
#     metadata = load_policy_metadata(policy_id)
#     metadata["enabled"] = bool(enabled)
#     meta_path = _policy_dir(policy_id) / METADATA_FILE
#     meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
#     return metadata



def set_policy_enabled(policy_id: str, enabled: bool) -> dict:
    """
    Updates the status of a policy in the policy_info collection.

    enabled=True  -> status = "selected"
    enabled=False -> status = "unselected"

    Returns the updated payload.
    """
    client = get_qdrant_client()

    status = "selected" if enabled else "unselected"

    # Update payload
    client.set_payload(
        collection_name=POLICY_INFO,
        payload={
            "status": status
        },
        points=Filter(
            must=[
                FieldCondition(
                    key='policy_id',
                    match=MatchValue(value=policy_id)
                )
            ]
        )
    )

    # Read back the updated payload
    points, _ = client.scroll(
        collection_name=POLICY_INFO,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key='policy_id',
                    match=MatchValue(value=policy_id)
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        raise ValueError(f"Policy '{policy_id}' not found.")

    payload = points[0].payload
    result =  {
            "id": payload['policy_id'],
            "name": payload['policy_id'],
            "country": payload['region'],
            "version": payload['version'],
            "enabled": payload['status'] == "selected"
        }
    
    return result


def delete_policy(policy_id: str) -> None:
    try:
        qdrant_delete_requirements(policy_id)
    except Exception as e:
        print(f"[policy_manager] Warning: Qdrant delete failed for '{policy_id}': {e}")


def _extract_text(file_path: Path, filename: str) -> str:
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        import pymupdf
        doc = pymupdf.open(file_path)
        text = "\n".join(page.get_text().strip() for page in doc)
        doc.close()
        return text

    if ext == ".docx":
        import docx
        d = docx.Document(str(file_path))
        return "\n".join(p.text for p in d.paragraphs)

    if ext in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8", errors="ignore")

    raise PolicyError(f"Unsupported policy file type: '{ext}'. Use PDF, DOCX, or TXT.")


def create_rag_chunks_pipeline(pdf_path: str):
    """Extracts text, table, and image chunks from a valid PDF."""
    start = time.perf_counter()


    print(f"Parsing document: {pdf_path}...")
    res = converter.convert(pdf_path)
    doc = res.document

    # Handle invalid or unparseable PDFs gracefully
    if doc is None:
        return [], "file is invalid"

    chunks = []
    
    # Hierarchy State Trackers
    current_h1 = "Main Document"
    current_h2 = ""
    current_h3 = ""
    
    # Sequential Paragraph Buffers
    text_buffer = []
    page_buffer = set()
    
    def flush_text_buffer():
        if text_buffer:
            hierarchy_str = f"{current_h1} > {current_h2} > {current_h3}".strip(" > ")
            section_name = current_h3 or current_h2 or current_h1
            text = section_name + "\n" + "\n\n".join(text_buffer)

            chunks.append({
                "type": "text_section",
                "text": text,
                "metadata": {
                    "file_name": os.path.basename(pdf_path),
                    "section_name": section_name,
                    "hierarchy": hierarchy_str,
                    "pages": sorted(list(page_buffer)),
                }
            })
            text_buffer.clear()
            page_buffer.clear()

    # Iterate through items in sequential reading order
    for item, level in doc.iterate_items():
        prov = getattr(item, "prov", None)
        page_num = prov[0].page_no if prov else 1

        # --- PATH A: TEXT & HEADINGS ---
        if isinstance(item, TextItem):
            text_content = item.text.strip()
            if not text_content:
                continue

            if item.label == DocItemLabel.SECTION_HEADER or item.label == DocItemLabel.TITLE:
                flush_text_buffer()

                if text_content.startswith("# ") or re.match(r"^\d+\.\s", text_content):
                    current_h1 = text_content.replace("#", "").strip()
                    current_h2, current_h3 = "", ""
                elif text_content.startswith("## "):
                    current_h2 = text_content.replace("##", "").strip()
                    current_h3 = ""
                else:
                    current_h3 = text_content.strip()
                
                page_buffer.add(page_num)

            else:
                text_buffer.append(text_content)
                page_buffer.add(page_num)

        
    # Final flush for trailing paragraphs
    flush_text_buffer()
    end = time.perf_counter()

    print("=="*30)
    print(f"chunks from pdf in: {end - start:.2f} seconds")
    print(f"{len(chunks)}")
    print("=="*30)

    return chunks

def create_policy_from_upload(
    file_path: Path,
    original_filename: str,
    name: str,
    category: str = "Custom",
    country: str = "Global",
    description: str = "",
    version: str = "",
) -> dict:
    """
    Admin -> Upload document -> Extract text -> Generate metadata ->
    Store policy -> Ready for analysis. No application code changes,
    no deployment.
    """
    # text = _extract_text(file_path, original_filename)
    # if not text.strip():
    #     raise PolicyError(f"No extractable text found in '{original_filename}'.")

    chunks = create_rag_chunks_pipeline(pdf_path=file_path)

    # raw = call_qwen(extraction_prompt, text)
    # requirements = parse_json_response(raw)

    # if not isinstance(requirements, dict) or not requirements:
    #     raise PolicyError(
    #         "Could not extract structured requirements from this document. "
    #         "Try a document with clearer, more explicit obligations."
    #     )
    # # normalize: ensure every value is a list of strings
    # clean_requirements = {}
    # for k, v in requirements.items():
    #     key = _slugify(str(k))
    #     if isinstance(v, list):
    #         clean_requirements[key] = [str(item) for item in v]
    # if not clean_requirements:
    #     raise PolicyError("Extracted requirements were empty after validation.")

    # policy_id = _unique_policy_id(_slugify(name or original_filename))
    policy_id = name
    # pdir = _policy_dir(policy_id)
    # pdir.mkdir(parents=True, exist_ok=True)

    # ext = Path(original_filename).suffix.lower()
    # shutil.copy(file_path, pdir / f"source{ext}")

    # metadata = {
    #     "id": policy_id,
    #     "name": name or original_filename,
    #     "version": version or str(datetime.now(timezone.utc).year),
    #     "category": category or "Custom",
    #     "description": description or f"Policy uploaded from '{original_filename}'.",
    #     "country": country or "Global",
    #     "enabled": True,
    #     "uploaded_at": datetime.now(timezone.utc).isoformat(),
    #     "total_sections": _total_sections(clean_requirements),
    #     "source": "upload",
    # }

    # (pdir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # (pdir / REQUIREMENTS_FILE).write_text(json.dumps(clean_requirements, indent=2), encoding="utf-8")
    try:
        qdrant_save_requirements(policy_id, chunks=chunks)
        s = policy_info(policy_id=policy_id,region=country,version=version)
        a = policy_aliginment(policy_name=policy_id)


        print(s)
        print(a)
    except Exception as e:
        print(f"[policy_manager] Warning: Qdrant sync failed for '{policy_id}': {e}")

    return f"{policy_id} uploaded successfully.."
