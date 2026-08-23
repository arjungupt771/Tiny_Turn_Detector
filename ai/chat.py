from gbvh_test import (
    get_chat_session,
    update_chat_history,
)
from ai.llm_client import call_qwen, policy_checker, policy_aligned_score, final_policy_aligned_report, general_answer
import pymupdf

import os
from storage.retriever import search_requirements, querys_relevent_chunks,policy_aligned_data, available_policies
from policy.manager import get_active_policies
from services.pdf_service import generate_pdf
from services.redis_service import get_messages, save_messages



# SYSTEM_PROMPT = """
# You are an expert GBVH Compliance Assistant.

# You MUST answer ONLY using:

# 1. Uploaded document
# 2. Generated compliance report
# 3. Evaluation JSON
# 4. Selected framework

# Never use outside knowledge.

# If the answer cannot be found, reply:

# "The uploaded report does not contain this information."

# Keep answers concise and professional.
# """

# new Prompt
SYSTEM_PROMPT = """
You are an expert GBVH Compliance Assistant.

You MUST answer ONLY using:

1. Uploaded document
2. Generated compliance report
3. Evaluation JSON
4. Selected framework
5. Relevant policy requirements retrieved for this specific question
   (these may come from any currently enabled framework, not just the
   document's best-match framework -- clearly attribute which policy
   a requirement came from when citing it)

Never use outside knowledge.

If the answer cannot be found, reply:

"The uploaded report does not contain this information."

Keep answers concise and professional.
"""



def _search_active_requirements(query: str, top_k: int=5):
    active_ids = {p["id"] for p in get_active_policies()}
    if not active_ids:
        return []
    results =  search_requirements(query, top_k=top_k*3)
    filtered = [r for r in results if r["policy_id"] in active_ids]
    return filtered[:top_k]


def build_prompt(session, question):

    history = ""

    for message in session["history"][-8:]:

        history += f'{message["role"].upper()}: {message["content"]}\n'

    documents_block = ""

    for filename in session["documents"]:
        alignment = session["alignment_scores"].get(filename)
        policy = session["policies"].get(filename, {})
        frameworks = session.get("frameworks", {}).get(filename, [])
        frameworks_line = ", ".join(
            f"{f.get('display_name')}: {f.get('alignment')}%"
            for f in frameworks if f.get("alignment") is not None
        )

        documents_block += f"""
------------------------
DOCUMENT: {filename}
BEST-MATCH FRAMEWORK: {policy.get("display_name")}
OVERALL ALIGNMENT (avg across all evaluated frameworks): {alignment if alignment is not None else "N/A"}%
ALL FRAMEWORK SCORES: {frameworks_line or "N/A"}

EVALUATION (best-match framework)
{session["evaluations"].get(filename)}

REPORT
{session["reports"].get(filename)}
------------------------
"""

    # --- NEW: pull requirements semantically matched to this question ---
    relevant_reqs = _search_active_requirements(question, top_k=5)
    if relevant_reqs:
        requirements_block = "\n".join(
            f"- [{r['policy_id']} / {r['category']}] {r['requirement']} (match: {r['score']:.2f})"
            for r in relevant_reqs
        )
    else:
        requirements_block = "No closely matching policy requirements found."

    prompt = f"""
{SYSTEM_PROMPT}

========================

OVERALL ORGANIZATIONAL SUMMARY

{session["overall_summary"]}

========================

PER-DOCUMENT DETAIL (use this to compare documents, find the lowest/highest
scoring document, or answer questions about one specific document)

{documents_block}

========================

RELEVANT POLICY REQUIREMENTS (semantically matched to the user's question,
across all enabled frameworks -- use these for precise citations)

{requirements_block}

========================

CHAT HISTORY

{history}

========================

USER QUESTION

{question}

"""

    return prompt


def answer_question(session_id, question,file_name):

    # session = get_chat_session(session_id)

    # if session is None:

    #     return {
    #         "success": False,
    #         "message": "Invalid session. Please start a new analysis."
    #     }

    # prompt = build_prompt(session, question)

    # print("x-x"*30)
    # print(f"answer_question file_name: {file_name}")
    # print("x-x"*30)

    # print("x-x"*30)
    # print(f"answer_question: {session_id}")
    # print("x-x"*30)

    general = general_answer(session_id=session_id,query=question)

    if general != "None":

        return {
                "success": session_id,
        
                "answer": general}

    policies = available_policies()

    print("-="*30)
    print(policies)
    print("-="*30)

    check = policy_checker(query=question,available_policies=policies)

    print("x-"*30)
    print(check)
    print("x-"*30)

    if check['is_policy_checker'] == True:

        if check['policy_name'] == None:

            answer = "*Please re-enter your query using one of the available frameworks :* "
            answer = answer + "\n" + "\n ".join(policies)

            messages = get_messages(session_id=session_id)
                
            messages.append({"role":"user", "content":question})

            messages.append({"role":"assistant","content":answer})
            save_messages(session_id=session_id,messages=messages)

            return {
                "success": session_id,
                "answer": answer,
                "gen_report": {
                    "path": None,
                    "name": None}}

        
        aligned_data = policy_aligned_data(policy_name=check['policy_name'])
        
        # print("x-"*30)
        # print(aligned_data)
        # print("x-"*30)

        # for i in os.listdir("/tmp/upload_report"):
        #     file = i.split("<>")
        #     if file[0] == session_id:
        #         file_name=file[1]
        #         file_path = i
            
        

        print("=="*30)
        print(file_name, type(file_name))
        print("=="*30)

        if file_name == "":
            files = os.listdir("/tmp/upload_report")
            if files == []:
                file_name = "no file present"
            for i in files:
                if i.split("<>")[0] == session_id:
                    file_name = i
                    print("=="*30)
                    print("if file is already there.")
                    print(file_name, type(file_name))
                    print("=="*30)  
                else:
                    file_name = "no file present"

        if file_name == "no file present":
            return {
                "success": True,
                "answer": "Please upload an document.",
                "gen_report": {
                            "path": None,
                            "name": None}}

        # file_path = os.path.join("/tmp/upload_report",os.listdir("/tmp/upload_report")[-1])
        file_path = os.path.join("/tmp/upload_report",file_name)

        doc = pymupdf.open(file_path)
        text = "\n".join(page.get_text().strip() for page in doc)
        doc.close()

        res = policy_aligned_score(aligned_data=aligned_data,text=text)

        answer = final_policy_aligned_report(user_query=question,res=res,session_id=session_id)

        new_file_name = file_name.split("<>")[1]
        gen_file_name = f"AI_generated_{new_file_name}"

        overall_pdf_path = generate_pdf(answer, file_name=gen_file_name) 

        # print("+1"*30)
        # print(f"report is ready to download: {overall_pdf_path}")
        # print("+1"*30)

        return {
        
                "success": session_id,
        
                "answer": answer,

                "gen_report": {
                    "path": overall_pdf_path,
                    "name": gen_file_name}
        
            }

    data = querys_relevent_chunks(query=question)

    retrieved_chunks = " ".join(data)

    prompt = f""" 
    You are a retrieval-based AI assistant.

    You will be provided with a set of retrieved document chunks enclosed within <context> tags.

    Rules:
    1. Answer the user's question using ONLY the information contained in the provided chunks.
    2. Do NOT use your own knowledge, assumptions, or external information.
    3. If the answer cannot be found completely or confidently in the provided chunks, respond with:
    "I couldn't find enough information in the provided documents to answer that question."
    4. Do not infer, speculate, or fill in missing details.
    5. If multiple chunks contain relevant information, combine them into a single coherent answer while remaining faithful to the source.
    6. If the chunks contain conflicting information, mention the conflict instead of choosing one.
    7. Do not mention or reference chunk numbers unless explicitly requested.
    8. Keep the answer concise, accurate, and grounded in the provided context.

    <context>
    {retrieved_chunks}
    </context>

    """

    try:
        answer = call_qwen(
            session_id,
            prompt,
            question
        )
    except Exception as e:
        return {
            "success": False,
            "message": "The assistant is temporarily unavailable. Please try again in a moment.",
            "gen_report": {
                    "path": None,
                    "name": None}
        }

    # update_chat_history(
    #     session_id,
    #     "user",
    #     question
    # )

    # update_chat_history(
    #     session_id,
    #     "assistant",
    #     answer
    # )

    return {

        "success": True,

        "answer": answer,
        "gen_report": {
                    "path": None,
                    "name": None}

    }