from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, re, json

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="SAP Note 3404390 Business Place Table Assessment & Remediation")

# ===== Table Replacement Map =====
TABLE_MAPPING = {
    "J_1BBRANCH": {"source": "P_BusinessPlace"}
}

SELECT_RE = re.compile(
    r"""(?P<full>
            SELECT\s+(?:SINGLE\s+)?        
            (?P<fields>[\w\s,~\*]+)        
            \s+FROM\s+(?P<table>\w+)       
            (?P<middle>.*?)                
            (?:
                (?:INTO\s+TABLE\s+(?P<into_tab>[\w@()\->]+))
              | (?:INTO\s+(?P<into_wa>[\w@()\->]+))
            )
            (?P<tail>.*?)
        )\.""",
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)
GENERIC_USAGE_RE = re.compile(
    r"""\b(?P<stmt>
            JOIN\s+
            |TABLES:\s+
            |TABLES\s+
            |(TYPE|LIKE)\s+TABLE\s+OF\s+
            |(TYPE|LIKE)\s+
        )
        (?P<table>\w+)
    """,
    re.IGNORECASE | re.VERBOSE
)

class Finding(BaseModel):
    snippet: str
    suggestion: str

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = ""
    code: Optional[str] = ""
    findings: List[Finding] = []

def remediation_comment(table: str) -> str:
    return f"TODO: {table.upper()} is obsolete in S/4HANA (SAP Note 3404390). Use released CDS view {TABLE_MAPPING[table.upper()]['source']} instead."

def remediate_select(sel_text: str, table: str) -> str:
    new_table = TABLE_MAPPING[table.upper()]["source"]
    comment = remediation_comment(table)
    return f"SELECT * FROM {new_table}.\n* {comment}"

def remediate_nonselect(stmt_prefix: str, table: str) -> str:
    new_table = TABLE_MAPPING[table.upper()]["source"]
    comment = remediation_comment(table)
    replaced = re.sub(rf"\b{table}\b", new_table, stmt_prefix + table, flags=re.IGNORECASE)
    return f"{replaced}\n* {comment}"

def parse_findings(unit: Unit) -> List[Finding]:
    code = unit.code or ""
    findings: List[Finding] = []

    for m in SELECT_RE.finditer(code):
        table = m.group("table")
        if table and table.upper() in TABLE_MAPPING:
            snippet = m.group("full").strip() + "."
            suggestion = remediate_select(snippet, table)
            findings.append(Finding(snippet=snippet, suggestion=suggestion))

    for m in GENERIC_USAGE_RE.finditer(code):
        table = m.group("table")
        stmt_prefix = m.group("stmt")
        if table and table.upper() in TABLE_MAPPING:
            snippet = (stmt_prefix + table).strip()
            suggestion = remediate_nonselect(stmt_prefix, table)
            findings.append(Finding(snippet=snippet, suggestion=suggestion))
    return findings

# ===== LLM Prompt Setup =====
SYSTEM_MSG = """
You are a senior ABAP reviewer familiar with SAP Note 3404390. Output ONLY JSON as response.

For every finding[].snippet, write a bullet point that:
- Displays the exact offending code
- Explains the action needed using .suggestion text
- Do not omit any snippet; all must be covered. Do not reference by index; always show full code inline

Return valid JSON:
{{
  "assessment": "<paragraph summary of J_1BBRANCH usages and S/4HANA impact>",
  "llm_prompt": "<bulleted action items, one per finding, each with code and action>"
}}
""".strip()

USER_TEMPLATE = """
Unit metadata:
Program: {pgm_name}
Include: {inc_name}
Unit type: {unit_type}
Unit name: {unit_name}

ABAP code context (optional):
{code}

findings (JSON list of findings, each with .snippet and .suggestion for J_1BBRANCH usages):
{findings_json}

Instructions:
1. Write a 1-paragraph assessment summarizing J_1BBRANCH-related S/4HANA risks in human language.
2. Write a llm_prompt field: for each finding, add a bullet with:
   - The exact code (snippet)
   - The fix required (from suggestion)
   - Do NOT compress, omit or refer by index; always display code inline

Return ONLY valid JSON as above.
""".strip()

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_TEMPLATE),
])
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
parser = JsonOutputParser()
chain = prompt | llm | parser

def llm_assess_and_prompt(unit: Unit) -> Optional[Dict[str, str]]:
    findings_json = json.dumps([f.model_dump() for f in (unit.findings or [])], ensure_ascii=False, indent=2)
    if not unit.findings:
        return None
    try:
        return chain.invoke({
            "pgm_name": unit.pgm_name,
            "inc_name": unit.inc_name,
            "unit_type": unit.type,
            "unit_name": unit.name or "",
            "code": unit.code or "",
            "findings_json": findings_json,
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

@app.post("/assess-businessplace-migration")
async def assess_businessplace_migration(units: List[Unit]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for u in units:
        # Detect findings (offending ABAP usages)
        u.findings = parse_findings(u)
        if not u.findings:
            continue  # negative scenario: don't return anything
        llm_out = llm_assess_and_prompt(u)
        if not llm_out:
            continue
        obj = u.model_dump()
        obj.pop("findings", None)
        obj["assessment"] = llm_out.get("assessment", "")
        prompt_out = llm_out.get("llm_prompt", "")
        if isinstance(prompt_out, list):  # normalize if LLM returns list
            prompt_out = "\n".join(str(x) for x in prompt_out if x is not None)
        obj["llm_prompt"] = prompt_out
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}