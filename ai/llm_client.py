
import json
import time
import requests
import ollama
import json
from services.redis_service import get_messages, save_messages



generate_url = 'https://profiles-sorry-wallace-taken.trycloudflare.com/generate'

url = "https://profiles-sorry-wallace-taken.trycloudflare.com/pipeline_compatable"

policy_generate_url = "https://profiles-sorry-wallace-taken.trycloudflare.com/policy_generate"




def general_answer(session_id:str,query:str):


    p = """
        You are a query router.

        ---------------------
        STEP 1: DECISION LOGIC
        ---------------------

        Your task is to decide whether the user query is:
        1. GENERAL → answer directly
        2. None

        ---------------------
        GENERAL QUERIES
        ---------------------

        Answer directly if the query is:
        - a greeting or casual message
        - general knowledge
        - translation
        - spelling correction
        - meaning of normal words (not related to codes or classification)

        ---------------------
        POLICY EXCLUSION RULE
        ---------------------

        If the query is related to ANY policy, guideline, compliance document, or regulation (e.g., workplace policy, sexual harassment policy, HR guidelines, company rules), IMMEDIATELY return: None

        ---------------------
        PRIORITY RULE
        ---------------------

        If the query is policy-related, OR if there is any doubt, return: None

        ---------------------
        OUTPUT RULE
        ---------------------

        - GENERAL → return only the answer
        - None
        - No explanation
        - No extra text

        FINAL ANSWER:
        """


    messages = get_messages(session_id=session_id)

    messages.insert(0, {"role":"system","content":p})

    messages.append({"role":"user", "content":query})

    # print(messages[-1])

    payload = json.dumps(list(messages))

    payload = {"messages":payload}


    # payload = {
    #     "prompt": p,
    #     "query": query
    # }

   
    response = requests.post(url, json=payload)

    if response.json()['response'].lower() != "none":
        messages.append({"role":"assistant","content":response.json()['response']})
        save_messages(session_id=session_id,messages=messages)

    
    return response.json()['response']

    

# def policy_checker(query:str):
#     query_prompt = """
#     You are a query classifier.

#     Your task is to determine whether the user's query is asking about:
#     - policy alignment,
#     - policy compliance,
#     - policy conformance,
#     - policy consistency,
#     - policy comparison against another policy,
#     - whether a policy follows, satisfies, adheres to, or meets the requirements of another policy or framework.

#     Return:
#     - True → if the query is asking whether one policy is aligned with, compliant with, follows, satisfies, conforms to, or can be evaluated against another policy, standard, framework, guideline, convention, regulation, or requirements.
#     - None → for every other type of query.

#     Rules:
#     - Return exactly one value: True or None.
#     - Do not return explanations.
#     - Do not return JSON.
#     - Do not return Markdown.
#     - Consider synonyms such as align, alignment, comply, compliance, conform, conformance, adhere, adherence, follow, satisfy, consistency, benchmark, compare against, map to, evaluate against, assess against, gap analysis, coverage, crosswalk, match requirements, and compatibility.
#     - If the query is simply asking to summarize, explain, extract, rewrite, translate, or answer questions about a policy, return None.
#     - If the query asks whether one policy meets the requirements of another policy or framework, return True.

#     """

#     payload = {
#         "prompt": query_prompt,
#         "query": query
#     }

#     response = requests.post(generate_url, json=payload)

#     return response.json()['response']

def policy_checker(query: str, available_policies: list):
    policies_text = "\n".join(
        f"- {policy}" for policy in available_policies
    )

    query_prompt = f"""
You are a policy query classifier.

TASK:
Analyze the user query and determine whether the user wants to check,
compare, assess, align, comply, conform, or evaluate a policy against
another policy, framework, standard, guideline, convention, regulation,
or requirement.

AVAILABLE POLICIES:
{policies_text}

IMPORTANT POLICY SELECTION RULE:
- You MUST select the policy only from AVAILABLE POLICIES.
- NEVER create, invent, modify, or hallucinate a policy name.
- If the user mentions a policy name that matches an available policy,
  select that policy.
- Matching should be case-insensitive.
- Minor spelling mistakes, abbreviations, punctuation differences,
  and common variations may be matched to the closest available policy.
- If the user mentions a policy that is NOT present in AVAILABLE POLICIES,
  return policy_name as null.
- If multiple available policies could match, select the most relevant one.
- If no reliable match exists, return null.

CLASSIFICATION:

Return true when the user asks about:
- policy alignment
- policy compliance
- policy conformance
- policy consistency
- policy comparison
- policy benchmarking
- policy mapping
- policy crosswalk
- gap analysis
- coverage against a policy
- whether a policy follows another policy
- whether a policy satisfies requirements
- whether a policy adheres to a framework
- whether a policy meets a standard
- whether a policy is compatible with another policy/framework
- assessing one policy against another policy/framework

Return null when the user only wants:
- policy summary
- policy explanation
- policy details
- policy extraction
- policy translation
- policy rewriting
- general questions about a policy
- information about a policy without comparison/alignment/compliance checking

KEYWORDS THAT CAN INDICATE POLICY CHECKING:
align, alignment, aligned,
comply, compliance, compliant,
conform, conformance, conforms,
adhere, adherence, adheres,
follow, follows,
satisfy, satisfies,
meet, meets,
consistent, consistency,
compare, comparison,
benchmark,
map, mapping,
crosswalk,
gap analysis,
coverage,
evaluate against,
assess against,
check against,
measure against,
match requirements,
compatible with

POLICY NAME EXTRACTION:
- Extract the target/reference policy from the user query.
- The target policy is the policy whose requirements are being used
  for comparison or evaluation.
- Select the policy from AVAILABLE POLICIES only.
- Do not return the user's own/internal policy unless it is the
  reference policy being evaluated against.
- If no available policy is clearly identified, return null.

OUTPUT:
Return ONLY valid JSON.

For a policy-checking query:
{{
  "is_policy_checker": true,
  "policy_name": "EXACT NAME FROM AVAILABLE POLICIES"
}}

For a non-policy-checking query:
{{
  "is_policy_checker": null,
  "policy_name": null
}}

For a policy-checking query where the requested policy is unavailable:
{{
  "is_policy_checker": true,
  "policy_name": null
}}

STRICT RULES:
1. Output JSON only.
2. No Markdown.
3. No explanations.
4. No additional fields.
5. Never invent a policy.
6. policy_name MUST be copied from AVAILABLE POLICIES.
7. If no reliable policy match exists, policy_name MUST be null.

USER QUERY:
"""

    payload = {
        "prompt": query_prompt,
        "query": query
    }

    response = requests.post(generate_url, json=payload)

    result = response.json()["response"]
    
    try:
        result = json.loads(result)
        return result
    except:
        return result


def policy_aligned_score(aligned_data:dict,text:str):

    # p = """
    #     #     You are a compliance evaluator.

    #     #     Input:
    #     #     1. REQUIREMENTS_JSON
    #     #     2. REPORT_TEXT

    #     #     Evaluate REPORT_TEXT only against REQUIREMENTS_JSON.
    #     #     Do not use external knowledge.

    #     #     For each category:
    #     #     - Match requirements semantically.
    #     #     - Calculate:
    #     #     Category Alignment = (sum of requirement scores / total requirements) × 100
    #     #     - Extract only the requirements that are addressed (score > 0).
    #     #     - Combine evidence into a single concise summary instead of repeating it.
    #     #     - Do not list every missing requirement.

    #     #     Calculate:
    #     #     Overall Alignment = average of all category alignment percentages.

    #     #     Return ONLY valid JSON in this format:

    #     #     {
    #     #     "overall_alignment": 78,
    #     #     "categories": [
    #     #         {
    #     #         "category": "...",
    #     #         "alignment": 80,
    #     #         "covered": [
    #     #             "requirement 1",
    #     #             "requirement 2"
    #     #         ],
    #     #         "evidence": "Brief combined evidence from the report."
    #     #         }
    #     #     ],
    #     #     "top_gaps": [
    #     #         "gap 1",
    #     #         "gap 2",
    #     #         "gap 3"
    #     #     ]
    #     #     }
    #      """
    
    p = """  You are an expert policy compliance evaluator.

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

    payload = {"prompt":p,
                    "query":f"REQUIREMENTS_JSON:\n{aligned_data}\n\nREPORT_TEXT:\n{text}"}
    
    
    response = requests.post(generate_url,json=payload)


    res = response.json()['response']

    return res


def final_policy_aligned_report(user_query:str,res,session_id:str):

    # final_output_prompt = """You are an expert report analysis assistant.

    #     You will receive a JSON containing the results of a compliance assessment.

    #     Analyze the JSON and generate a concise, professional report.

    #     Your report should include:

    #     1. **Overall Summary**
    #     Briefly describe the overall alignment and key findings.

    #     2. **Strengths**
    #     Highlight the categories and requirements with high compliance.

    #     3. **Gaps**
    #     Identify missing or partially addressed requirements.

    #     4. **Recommendations**
    #     Provide prioritized, actionable recommendations to improve compliance.

    #     5. **Conclusion**
    #     Summarize the organization's current compliance status and the most important next steps.

    #     Guidelines:

    #     - Base your analysis only on the provided JSON.
    #     - Do not invent or assume information.
    #     - Group similar findings instead of repeating individual requirements.
    #     - Keep the report concise, objective, and professional.
    #     - If evidence is available in the JSON, use it to support your observations.
    #     - If overall alignment is below 60%, emphasize critical gaps.
    #     - If overall alignment is above 80%, emphasize strengths while noting remaining improvements.
    #     - Use the exact compliance data and percentages provided in the JSON.
    #     - Do not calculate or infer values that are not present in the JSON.

    #     Return the report in Markdown format with clear section headings.

    #     Output Requirements:

    #     - Return ONLY the report.
    #     - Do NOT include placeholders such as:
    #     - "Prepared by"
    #     - "Author"
    #     - "Date"
    #     - "Contact Information"
    #     - "Company"
    #     - "Appendix"
    #     - "Disclaimer"
    #     - "Notes"
    #     - "Feel free to..."
    #     - "You can customize..."
    #     - "Adjust as needed."
    #     - Do NOT add introductions or conversational remarks.
    #     - Do NOT add conclusions outside the requested "Conclusion" section.
    #     - Do NOT generate template sections.
    #     - Do NOT generate markdown separators such as "---".
    #     - Do NOT use placeholder text such as "[Your Name]", "[Company]", "[Date]", or similar.
    #     - End the response immediately after the Conclusion section.
    #     - Produce only the requested report with no extra commentary.
    #     """


    final_output_prompt = """
You are an expert compliance report writer.

You will receive a JSON containing the results of a compliance assessment.

Your task is to transform the provided assessment JSON into a concise, professional compliance report.

IMPORTANT:
The assessment JSON is the ONLY source of truth.

Do not perform a new compliance assessment.
Do not introduce new requirements.
Do not infer new gaps, strengths, controls, or recommendations.
Do not use external knowledge.
Do not evaluate the original report or policy independently.
Only summarize and organize the findings already present in the assessment JSON.

The report must contain exactly these sections:

1. Overall Summary

Briefly summarize:
- the overall alignment percentage provided in the assessment JSON
- the major strengths
- the major compliance gaps

Use the exact alignment value provided in the JSON.

2. Strengths

Highlight categories and requirements that have:
- score = 100, or
- high category alignment.

Use only findings supported by the assessment JSON.

Group related requirements into concise themes rather than repeating every requirement.

3. Gaps

Identify the most important requirements or categories that are:
- score = 0, or
- score = 50.

Use the gaps and requirement-level findings provided in the assessment JSON.

Do not create additional gaps.

Group related gaps where appropriate.

4. Recommendations

Provide actionable recommendations derived directly from the identified gaps.

Each recommendation must address an actual gap present in the assessment JSON.

Do not recommend controls, policies, committees, audits, training, standards, or processes unless the recommendation is supported by a gap in the assessment JSON.

Prioritize recommendations according to the importance of the identified gaps.

5. Conclusion

Summarize the organization's compliance position using only the assessment findings.

Mention the most important strengths and remaining gaps.

Do not introduce new findings.

SCORING RULES:

- Use the exact overall alignment percentage provided in the JSON.
- Use the exact category alignment percentages provided in the JSON when mentioning them.
- Do not recalculate scores.
- Do not modify, reinterpret, or estimate percentages.
- Do not change a requirement's compliance status.
- Do not convert a partial score into full compliance or vice versa.

EVIDENCE:

- If evidence is included in the assessment JSON, use it when explaining findings.
- Do not invent evidence.
- Do not claim that an organization has a control unless the assessment JSON provides evidence for it.
- For a score of 0, describe the requirement as not sufficiently addressed rather than claiming that the organization definitively does not have the control.

REPORT STYLE:

- Concise
- Objective
- Professional
- Clear
- Evidence-based
- Suitable for a formal compliance assessment

Do not repeat the same finding across multiple sections unnecessarily.

If overall alignment is below 60%, emphasize the most significant gaps and remediation priorities.

If overall alignment is between 60% and 80%, provide a balanced discussion of strengths and gaps.

If overall alignment is above 80%, emphasize strengths while still identifying remaining gaps.

OUTPUT REQUIREMENTS:

Return ONLY the report in Markdown format.

Use these exact headings:

## Overall Summary
## Strengths
## Gaps
## Recommendations
## Conclusion

Do NOT include:

- Prepared by
- Author
- Date
- Contact Information
- Company
- Appendix
- Disclaimer
- Notes
- Sources
- References
- Feel free to...
- You can customize...
- Adjust as needed.
- Any other template or placeholder sections

Do NOT include:
- placeholder text such as [Your Name], [Company], [Date]
- introductions before "Overall Summary"
- commentary outside the requested sections
- markdown separators such as "---"

End the response immediately after the Conclusion section.
"""

    policy_aligned_json = res

    messages = get_messages(session_id=session_id)
    messages.insert(0,{"role":"system","content": final_output_prompt})

    messages.append({"role":'user',"content":policy_aligned_json})

    # print(messages[-1])

    payload = json.dumps(list(messages))

    payload = {"messages":payload}


    # payload = {"prompt":final_output,
    #             "query":query}


    response = requests.post(url,json=payload)

    res = response.json()['response']
    # remove the policy_aligned data from the messages
    _ = messages.pop()

    # add the user query in the messages
    messages.append({"role":"user","content":user_query})

    messages.append({"role":"assistant","content":res})
    save_messages(session_id=session_id,messages=messages)


    return res




def call_qwen(session_id:str,prompt, query):
    """
    Calls the LLM backend with basic retry/timeout handling so a single
    flaky request doesn't take down report generation, chat, or the
    AI copilot.
    """

    messages = get_messages(session_id=session_id)

    messages.insert(0, {"role":"system","content":prompt})

    messages.append({"role":"user", "content":query})

    # print(messages[-1])

    payload = json.dumps(list(messages))
    payload = {"messages":payload}

   
    # payload = {
    #     "prompt": prompt,
    #     "query": query
    # }

    response = requests.post(url, json=payload)

    messages.append({"role":"assistant","content":response.json()['response']})

    save_messages(session_id=session_id,messages=messages)

    return response.json()['response']


# def policy_alignment_data(policy_text:str):

#     prompt = """
#     You are a precise Policy Compliance Analyst AI. Your sole task is to analyze policy documents and generate a JSON compliance checklist to audit other internal policies for alignment.

#     ### RULES & CONSTRAINTS:
#     1. OUTPUT FORMAT: Respond STRICTLY in raw, valid JSON. 
#     - Do NOT use Markdown code blocks or ```json fences.
#     - Do NOT include intro, outro, explanations, or conversational text.
#     - Do NOT include trailing commas.
#     - Use double quotes for all keys and string values.
#     - The response must be directly parseable by Python's json.loads().

#     2. CONTENT EXTRACTION:
#     - Extract requirements ONLY from the provided reference policy text.
#     - Do NOT use external knowledge, assume details, or infer facts not present.
#     - Classify each requirement strictly as either "Mandatory" or "Recommended".
#     - Group criteria logically by section using snake_case field names.

#     ### REQUIRED JSON SCHEMA:
#     {
#     "policy_metadata": {
#         "reference_document": "<String: Exact policy title>",
#         "issuing_body": "<String: Issuing empty entity if name, not or organization stated string>",
#         "purpose": "<String: Purpose checklist compliance of the>"
#     },
#     "sections": [
#         {
#         "section_id": "<String: 'I', 'II', 'III' Section e.g., identifier,>",
#         "section_title": "<String: Section Title>",
#         "requirements": [
#             {
#             "id": "<String: 'DEF-01', 'FRM-01' 'POL-01', ID, Unique alphanumeric e.g.,>",
#             "item": "<String: Concise name of requirement>",
#             "description": "<String: Explicit condition in or policy rule stated text>",
#             "classification": "<String: Mandatory Recommended or>",
#             "target_key_phrase": "<String: Exact for from key keywords matching or phrase text>"
#             }
#         ]
#         }
#     ]
#     }

#     ### POLICY TEXT TO ANALYZE:
#     """
#     prompt = prompt + policy_text

#     response = ollama.generate(
#         model="qwen2.5:14b",
#         prompt=prompt,
#         format="json",
#         options=model_parameters
#     )

#     # output = json.loads(response["response"])


#     return response['response']

def extract_policy_compliance_requirements(policy_text: str):

#     prompt = """
# You are a Policy Compliance Analyst.

# Analyze the reference policy text and extract its explicit compliance requirements into the required JSON structure.

# Use only information from the provided policy. Do not use external knowledge, assumptions, or invented requirements.

# For each requirement:

# * Extract only explicit rules, obligations, standards, responsibilities, or recommendations.
# * Classify it as exactly "Mandatory" or "Recommended".
# * Use a concise description.
# * Set `target_key_phrase` to an exact phrase from the policy text.
# * Group requirements by their relevant policy section.
# * Do not duplicate requirements.

# Return only valid JSON matching the specified schema. Do not include explanations, summaries, Markdown, or any text outside the JSON.

# ========================
# REFERENCE POLICY TEXT
# ========================

# """ + policy_text
    prompt = """
    You are an expert Policy Compliance Requirement Extractor.
    
    Your task is to analyze the provided POLICY TEXT and extract the most
    accurate, complete, source-grounded, and evaluation-ready set of explicit
    requirements contained in the document.
    
    The extracted requirements will later be used to evaluate another document
    against this policy.
    
    The POLICY TEXT may come from any organization, company, HR department,
    government, international organization, UN, ILO, supplier framework,
    corporate policy, code of conduct, human rights policy, safety policy,
    privacy policy, ESG policy, governance policy, or other compliance-related
    document.
    
    You must remain DOMAIN-AGNOSTIC.
    
    Use ONLY the provided POLICY TEXT.
    
    Do NOT use:
    - external knowledge
    - legal knowledge
    - industry standards
    - assumptions
    - common practices
    - information from other policies
    - information from referenced documents
    - information from external frameworks unless explicitly incorporated into
      the current policy
    
    ============================================================
    1. CORE OBJECTIVE
    ============================================================
    
    Extract every genuine requirement that is explicitly established by the
    POLICY TEXT and that can reasonably be evaluated against another document.
    
    A requirement may be expressed as:
    
    - an obligation
    - a required action
    - a required behavior
    - a prohibition
    - a reporting duty
    - a legal or regulatory compliance requirement
    - a right, entitlement, freedom, or protection
    - An explicitly established, independently evaluatable organizational control or responsibility
    - a supplier/vendor/business-partner requirement
    - a governance or accountability responsibility
    - an explicit expectation
    - an explicit recommendation or encouragement
    - an explicit evaluatable standard
    
    Do NOT require words such as "must" or "shall".
    
    Determine whether the COMPLETE statement establishes an actionable or
    evaluatable requirement.
    
    ============================================================
    2. DOMAIN-AGNOSTIC EXTRACTION
    ============================================================
    
    Do not assume the document belongs to a particular policy type, company,
    industry, country, jurisdiction, or regulatory framework.
    
    The same extraction logic must work for:
    
    - HR policies
    - global policies
    - human rights policies
    - workplace policies
    - supplier policies
    - codes of conduct
    - anti-bribery policies
    - safety policies
    - privacy/security policies
    - ESG policies
    - governance policies
    - UN documents
    - ILO documents
    - other international standards or compliance frameworks
    
    Determine extractability solely from the language and structure of the
    provided POLICY TEXT.
    
    ============================================================
    3. WHAT TO EXTRACT
    ============================================================
    
    Extract explicit statements establishing:
    
    A. REQUIRED BEHAVIOR
    Example:
    "Employees must complete annual training."
    
    B. PROHIBITIONS
    Example:
    "Employees must not engage in discriminatory conduct."
    
    C. REPORTING OBLIGATIONS
    Example:
    "Employees must report suspected violations."
    
    D. RIGHTS / ENTITLEMENTS / PROTECTIONS
    Example:
    "Employees are entitled to fair treatment."
    
    E. ORGANIZATIONAL CONTROLS
    Example:
    "The organization conducts annual risk assessments."
    
    F. SUPPLIER / EXTERNAL-PARTY REQUIREMENTS
    Example:
    "Suppliers are expected to comply with these standards."
    
    G. GOVERNANCE / ACCOUNTABILITY
    Example:
    "A designated officer is responsible for overseeing the program."
    
    H. EXPLICIT RECOMMENDATIONS
    Example:
    "Organizations should conduct periodic reviews."
    
    Extract these only when the source itself establishes them.
    
    
    ============================================================
    EXCLUSION: SCOPE AND APPLICABILITY STATEMENTS
    ============================================================
    
    Do NOT extract a Scope or Applicability statement as a standalone
    requirement when it merely identifies who, where, or what the policy
    applies to.
    
    Example:
    
    "This Policy applies to all directors, officers, and employees..."
    
    This is contextual applicability information, NOT a standalone requirement.
    
    Use such information to understand the applicability of other extracted
    requirements, but do not create a requirement from it.
    
    Only extract the statement if the scope sentence itself establishes an
    action, obligation, prohibition, right, protection, control, or other
    evaluatable requirement.
    
    
    SCOPE SECTION HANDLING
    
    Preserve the actual section name "Scope" when it appears in the document.
    
    Do not automatically exclude an entire Scope section.
    
    Analyze its contents like any other section.
    
    However, a statement whose sole purpose is to identify who, what,
    where, or which entities the policy applies to is contextual
    applicability information, not an independently evaluatable requirement.
    
    Example:
    
    "This Policy applies to all directors, officers, and employees."
    
    Do NOT extract this as a requirement.
    
    However, if a Scope section contains an explicit obligation,
    prohibition, right, protection, control, expectation, or other
    evaluatable requirement, extract that requirement normally.
    
    Example:
    
    "Employees must comply with this policy when working at client sites."
    
    Extract this as a requirement.
    
    Therefore:
    - section name = preserve
    - applicability statement = do not extract as requirement
    - actionable requirement within Scope = extract
    
    ============================================================
    EXCLUSION: DESCRIPTIONS OF REFERENCED POLICIES
    ============================================================
    
    Do NOT extract a requirement merely because a sentence says that another
    policy, standard, code, or document:
    
    - prohibits something
    - addresses something
    - contains requirements
    - is embedded in contracts
    - establishes standards
    - provides guidance
    
    The contents of a referenced document must NOT be imported into the current
    policy.
    
    Example:
    
    "Our Supplier Standards of Conduct prohibits and addresses human rights
    abuses..."
    
    Do NOT convert this into:
    
    "Suppliers must comply with the Supplier Standards of Conduct."
    
    unless the CURRENT POLICY explicitly establishes that obligation.
    
    ============================================================
    EXCLUSION: ORGANIZATIONAL DESCRIPTIONS VS GOVERNANCE REQUIREMENTS
    ============================================================
    
    Do NOT extract a governance or organizational statement merely because it
    assigns or describes a role.
    
    Example:
    
    "The Chief Ethics and Compliance Officer spearheads the day-to-day
    operation of the program."
    
    This is an organizational description and should NOT be extracted unless
    the policy explicitly establishes a required action, accountability,
    reporting duty, oversight obligation, or control that can be evaluated.
    
    Similarly:
    
    "E&C program oversight responsibility is vested in the Audit Committee."
    
    Do NOT extract this merely as a requirement because it describes where
    responsibility resides.
    
    Extract governance statements only when they establish a concrete,
    evaluatable responsibility or required action.
    
    Example:
    
    "The E&C Officer must report compliance findings annually to the Audit
    Committee."
    
    This IS an extractable requirement.
    
    ============================================================
    4. COMMITMENTS, VALUES AND ASPIRATIONS
    ============================================================
    
    Do NOT extract a statement merely because it expresses:
    
    - a general commitment
    - a value
    - an aspiration
    - an intention
    - a belief
    - a general objective
    - a broad ethical principle
    
    Example:
    
    "We are committed to respecting human rights."
    
    Do NOT automatically extract this as a requirement.
    
    However, if a statement containing commitment language also contains a
    separate explicit obligation, expectation, right, protection, or control,
    extract the actionable portion.
    
    Do not discard genuine requirements merely because they appear inside a
    paragraph containing general commitment language.
    
    ============================================================
    5. NORMATIVE PRINCIPLES VS REQUIREMENTS
    ============================================================
    
    Some documents, particularly international frameworks and standards, may
    contain:
    
    - principles
    - rights
    - standards
    - objectives
    - aspirations
    - recommendations
    - normative statements
    
    Do NOT automatically convert every principle or normative statement into a
    Mandatory requirement.
    
    Extract a statement only when the source explicitly establishes it as:
    
    - an obligation
    - a prohibition
    - required behavior
    - a right or protection
    - an explicit expectation
    - a required control
    - an explicit recommendation
    - an explicit evaluatable standard
    
    Preserve the source's original level of obligation.
    
    Do NOT strengthen:
    
    "should" into "must"
    
    "encourages" into "requires"
    
    "aims to" into "must"
    
    "supports" into "requires"
    
    "recognizes" into "requires"
    
    ============================================================
    6. MANDATORY VS RECOMMENDED
    ============================================================
    
    Every extracted requirement MUST be classified as exactly:
    
    "Mandatory"
    
    or
    
    "Recommended"
    
    MANDATORY includes explicit:
    
    - obligations
    - prohibitions
    - required actions
    - required behaviors
    - reporting duties
    - rights/protections established by the policy
    - required organizational controls
    - explicit supplier requirements
    - clear policy expectations
    - explicit requirements expressed through equivalent wording
    
    RECOMMENDED includes only explicit:
    
    - recommendations
    - encouragements
    - suggestions
    - voluntary actions
    - "should"
    - "encourages"
    - "urges"
    - "recommends"
    
    Do NOT classify a requirement based solely on a keyword.
    
    Interpret the complete statement and its context.
    
    Words such as "expected to" may represent a Mandatory policy expectation
    when the surrounding context clearly establishes it as an applicable policy
    requirement.
    
    ============================================================
    7. SUBJECT PRESERVATION
    ============================================================
    
    Preserve the exact subject identified by the policy.
    
    Possible subjects may include:
    
    - employees
    - associates
    - directors
    - officers
    - managers
    - the organization
    - suppliers
    - vendors
    - contractors
    - business partners
    - clients
    - representatives
    - government bodies
    - states
    - employers
    - workers
    - other explicitly identified parties
    
    Do NOT change the subject.
    
    Do NOT assume that a requirement applying to one party applies to another.
    
    Do NOT automatically convert an organizational requirement into an
    employee requirement.
    
    Do NOT automatically convert a supplier requirement into a company
    requirement.
    
    ============================================================
    8. SCOPE AND APPLICABILITY
    ============================================================
    
    Preserve explicit scope and applicability.
    
    Examples include:
    
    - specific employees
    - specific business units
    - geographic regions
    - subsidiaries
    - suppliers
    - client sites
    - supply chains
    - specific activities
    - employment decisions
    - specific business processes
    - high-risk suppliers
    
    If a requirement applies only under a particular condition or circumstance,
    preserve that condition.
    
    Do NOT broaden or narrow the scope.
    
    ============================================================
    9. CONDITIONS
    ============================================================
    
    Preserve conditional language.
    
    Example:
    
    "If you know or suspect that a violation has occurred, report it."
    
    Do NOT convert this into:
    
    "Employees must always report violations."
    
    The extracted requirement must preserve the original condition.
    
    ============================================================
    10. TIMING AND FREQUENCY
    ============================================================
    
    Preserve all explicit timing and frequency information.
    
    Examples:
    
    - immediately
    - promptly
    - annually
    - periodically
    - during onboarding
    - before deployment
    - at least once per year
    - when applicable
    - upon discovery
    
    Do NOT remove timing or frequency because it may be critical to later
    compliance evaluation.
    
    ============================================================
    11. ATOMIC REQUIREMENTS
    ============================================================
    
    Each requirement must represent ONE independently evaluatable obligation,
    right, protection, control, recommendation, or prohibition.
    
    If a sentence contains multiple independent requirements, split them.
    
    Example:
    
    "Employees must complete training and report suspected violations."
    
    Extract:
    
    1. Employees must complete training.
    2. Employees must report suspected violations.
    
    However, do NOT split supporting examples or explanatory details from a
    single requirement.
    
    Example:
    
    "Employees must not engage in harassment, including sexual harassment and
    workplace bullying."
    
    Keep this as one requirement unless the source separately establishes the
    examples as independent requirements.
    
    ============================================================
    12. DO NOT OVER-EXTRACT
    ============================================================
    
    Do NOT extract:
    
    - definitions
    - background information
    - historical information
    - policy purpose
    - general values
    - general commitments
    - aspirations
    - objectives without actionable requirements
    - explanatory statements
    - document navigation text
    - page numbers
    - headers
    - footers
    - revision history
    - effective dates
    - policy metadata
    - document ownership information
    - consequences of violations
    - references to other documents by themselves
    
    ============================================================
    13. REFERENCED POLICIES AND FRAMEWORKS
    ============================================================
    
    Do NOT import requirements from another document.
    
    If the policy says:
    
    "See the Supplier Code for additional requirements."
    
    Do NOT extract the Supplier Code's requirements.
    
    Only extract requirements explicitly established by the CURRENT POLICY TEXT.
    
    Similarly, references to external frameworks such as:
    
    - UN
    - ILO
    - OECD
    - UNESCO
    - ISO
    - NIST
    - legislation
    - regulations
    - other standards
    
    do NOT automatically create requirements.
    
    Only extract a framework-based requirement when the CURRENT POLICY
    explicitly incorporates, requires, adopts, or directs compliance with that
    framework or its stated provisions.
    
    Do not retrieve or infer the external framework's contents.
    
    ============================================================
    14. ORGANIZATIONAL CONTROLS VS DESCRIPTIVE PRACTICES
    ============================================================
    
    Extract an organizational control when the policy explicitly establishes a
    concrete process, control, recurring activity, or operational mechanism
    that can be evaluated.
    
    Examples:
    
    "We conduct annual risk assessments."
    
    "Employees must complete mandatory annual training."
    
    "Suppliers are reassessed annually."
    
    Do NOT automatically extract purely descriptive statements such as:
    
    "We currently publish three reports."
    
    unless the policy establishes the activity as a required or recurring
    control.
    
    The distinction is:
    
    DESCRIPTIVE FACT:
    describes what currently exists.
    
    EXPLICIT CONTROL:
    establishes what the organization is required or expected to do.
    
    ============================================================
    15. PDF TEXT HANDLING
    ============================================================
    
    The POLICY TEXT may have been extracted from a PDF.
    
    PDF extraction may interrupt text because of:
    
    - page breaks
    - columns
    - tables
    - headers
    - footers
    - navigation elements
    - translation menus
    - formatting
    
    Reconstruct an interrupted sentence only when the continuation is directly
    supported by the surrounding text.
    
    Ignore obvious PDF artifacts such as:
    
    - page numbers
    - repeated headers/footers
    - translation language menus
    - navigation controls
    - document viewer text
    - unrelated metadata
    
    Do not allow PDF artifacts to become requirements.
    
    ============================================================
    16. DESCRIPTION
    ============================================================
    
    Each description must:
    
    - accurately represent the source
    - be concise
    - preserve the original meaning
    - preserve the subject
    - preserve scope
    - preserve conditions
    - preserve timing/frequency
    - preserve important limitations
    - contain no external interpretation
    
    Do NOT make the requirement stronger than the source.
    
    Do NOT weaken the requirement.
    
    Do NOT add words such as:
    
    - strictly
    - comprehensively
    - universally
    - continuously
    - legally
    - always
    
    unless explicitly supported by the source.
    
    Do NOT add implied obligations.
    
    ============================================================
    17. TARGET_KEY_PHRASE
    ============================================================
    
    The target_key_phrase MUST be an EXACT CONTIGUOUS phrase copied from the
    POLICY TEXT.
    
    It must:
    
    - appear verbatim in the source
    - not be paraphrased
    - not contain invented words
    - not combine text from separate locations
    
    Prefer a phrase that directly identifies the actionable portion of the
    requirement.
    
    
    
    Example:
    
    Source:
    "Employees must immediately report suspected violations."
    
    Valid:
    
    "must immediately report suspected violations"
    
    Invalid:
    
    "Employees should report violations"
    
    because it is not an exact source phrase.
    
    
    SOURCE VERIFICATION — ZERO INVENTION
    
    Every extracted requirement MUST be directly supported by the POLICY TEXT.
    
    The description MUST NOT introduce:
    - a new actor
    - a new obligation
    - a new prohibition
    - a new action
    - a new deadline
    - a new frequency
    - a new reporting duty
    - a new condition
    - a new scope
    
    If the source does not explicitly establish the action, do not create it.
    
    The target_key_phrase MUST be an exact contiguous substring of the
    POLICY TEXT.
    
    Before returning each requirement, verify that target_key_phrase occurs
    verbatim in the POLICY TEXT.
    
    If it does not occur verbatim, the requirement MUST be removed.
    
    Never construct a requirement by combining or transforming separate
    statements into a new obligation.
    
    ============================================================
    18. SECTION HANDLING
    ============================================================
    
    Use the most relevant section or heading explicitly present in the policy.
    
    Do NOT invent section names.
    
    If the policy contains sections, preserve their actual names.
    
    If a requirement appears within a subsection, use the most relevant actual
    heading.
    
    Do NOT create artificial sections such as:
    
    - General Requirements
    - Compliance
    - Legal Requirements
    
    unless those names actually appear in the policy.
    
    ============================================================
    19. DUPLICATES
    ============================================================
    
    Do not duplicate requirements.
    
    If the same requirement appears multiple times, extract it once unless the
    occurrences materially differ in:
    
    - subject
    - scope
    - condition
    - action
    - timing
    - applicability
    
    Do not create separate requirements for simple restatements.
    
    ============================================================
    20. AUDITABILITY TEST
    ============================================================
    
    Before extracting a candidate, ask BOTH:
    
    1. Does the policy explicitly establish something that a defined subject
       must, should, is expected to, is entitled to, or is responsible for doing,
       avoiding, maintaining, reporting, or receiving?
    
    AND
    
    2. Could an evaluator examine another document and objectively determine
       whether that requirement is fully covered, partially covered, or not
       covered?
    
    If either answer is NO, do not extract it.
    
    ============================================================
    21. TWO-PASS COMPLETENESS CHECK
    ============================================================
    
    After the initial extraction, review the ENTIRE POLICY TEXT again.
    
    Specifically search for missed:
    
    - obligations
    - prohibitions
    - reporting requirements
    - rights
    - protections
    - supplier requirements
    - organizational controls
    - training requirements
    - governance responsibilities
    - recurring processes
    - explicit expectations
    - recommendations
    - conditional requirements
    
    Do not stop after extracting only the most obvious bullet points.
    
    At the same time, remove any item that is actually:
    
    - background
    - a definition
    - a commitment
    - an aspiration
    - metadata
    - a reference
    - a consequence
    - an inferred obligation
    
    The objective is:
    
    MAXIMUM COVERAGE OF GENUINE REQUIREMENTS
    +
    MINIMUM INFERENCE
    
    ============================================================
    REQUIREMENT GATE — DO NOT EXTRACT CONTEXT AS REQUIREMENTS
    ============================================================
    
    Before adding ANY item to the output, determine whether the statement
    itself establishes something that can be evaluated.
    
    A statement is NOT a requirement merely because:
    
    - it appears inside a policy section;
    - the section is called "Scope", "Governance", "Overview", "Purpose",
      "Background", or another policy heading;
    - it contains words such as "applies", "responsible", "oversight",
      "zero tolerance", "prohibits", or "supports";
    - it describes an existing policy, program, role, structure, or practice.
    
    The statement itself must establish an explicit:
    
    - obligation
    - prohibition
    - required action
    - required behavior
    - reporting duty
    - right
    - entitlement
    - protection
    - required control
    - explicit expectation
    - explicit recommendation
    - evaluatable standard
    
    Otherwise, DO NOT extract it.
    
    IMPORTANT:
    
    A section heading does NOT determine whether its contents are
    requirements. Evaluate each statement independently.
    
    For example:
    
    "This Policy applies to all directors, officers, and employees."
    
    This is applicability context.
    DO NOT extract it.
    
    "The Chief Ethics and Compliance Officer is responsible for overseeing
    the program."
    
    If this merely describes organizational structure, DO NOT extract it.
    
    "The Chief Ethics and Compliance Officer must submit an annual report to
    the Audit Committee."
    
    This establishes an evaluatable action.
    EXTRACT it.
    
    "Cognizant has zero tolerance for human rights abuses."
    
    This expresses a policy position.
    Do NOT automatically convert it into a requirement.
    
    "Suppliers must not engage in human rights abuses."
    
    This establishes an explicit prohibition.
    EXTRACT it.
    
    "Our Supplier Standards of Conduct prohibits and addresses human rights
    abuses."
    
    This describes another policy/document.
    DO NOT extract it unless the CURRENT POLICY explicitly establishes an
    obligation concerning that document.
    
    The presence of a statement in a policy does not by itself make the
    statement a compliance requirement.
    
    ============================================================
    22. FINAL VALIDATION
    ============================================================
    
    Before returning the JSON, verify every requirement:
    
    1. Is it explicitly supported by the POLICY TEXT?
    2. Did I avoid external knowledge?
    3. Did I avoid assumptions?
    4. Is the meaning preserved?
    5. Is the subject preserved?
    6. Is the scope preserved?
    7. Are conditions preserved?
    8. Is timing/frequency preserved?
    9. Is it atomic?
    10. Is it independently evaluatable?
    11. Is target_key_phrase exact and contiguous?
    12. Is Mandatory/Recommended justified?
    13. Is it not merely a definition?
    14. Is it not merely background?
    15. Is it not merely a commitment?
    16. Is it not merely a referenced document?
    17. Is it not merely metadata?
    18. Is it not merely a consequence?
    19. Is it not a duplicate?
    20. Is it not an inferred obligation?
    
    If any answer indicates that the item is not a genuine requirement, remove
    or correct it.
    
    ============================================================
    23. OUTPUT FORMAT
    ============================================================
    
    Return ONLY valid JSON.
    
    Use EXACTLY this schema:
    
    {
      "policy_requirements": [
        {
          "section": "string",
          "requirements": [
            {
              "description": "string",
              "target_key_phrase": "string",
              "compliance_level": "Mandatory | Recommended"
            }
          ]
        }
      ]
    }
    
    Do NOT add any other fields.
    
    Do NOT add:
    
    - IDs
    - subjects
    - scope fields
    - requirement types
    - evidence
    - scores
    - explanations
    - reasoning
    - summaries
    - comments
    - confidence scores
    - metadata
    
    Subject, scope, conditions, timing, and applicability must be incorporated
    into the description when necessary.
    
    Return nothing except the JSON.
    
    ============================================================
    POLICY TEXT
    ============================================================
    
    """ + policy_text

    # response = ollama.generate(
    #     model="qwen2.5:14b",
    #     prompt=prompt,
    #     format="json",
    #     options={
    #         "temperature": 0.0,
    #         "top_k": 1,
    #         "top_p": 1.0,
    #         "repeat_penalty": 1.0,
    #         "seed": 42,
    #         "num_ctx": 16384
    #     }
    # )

    # output = json.loads(response["response"])

    payload = {"prompt":prompt,
               "query":""}

    res = requests.post(policy_generate_url,json=payload)

    output = json.loads(res.json()["response"])

    return output  # res.json()['response'] # output

def parse_json_response(text: str):
    """
    Best-effort parse of JSON the model returns, tolerant of markdown
    code fences and stray leading/trailing text.
    """
    if not text:
        return None

    cleaned = text.strip()

    if cleaned.startswith("```"):
        import re
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None
