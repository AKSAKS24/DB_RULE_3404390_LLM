from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, re, json

# ---- Env setup ----
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# LangChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="SAP Note 3404390 Business Place Table Assessment & Remediation")

# ===== Table Replacement Map =====
TABLE_MAPPING = {
    "J_1BBRANCH": {"source": "P_BusinessPlace"}
}

# ===== Regex Patterns =====
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
# Non-SELECT usages
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

# ===== Models =====
class SelectItem(BaseModel):
    table: str
    target_type: Optional[str] = None
    target_name: Optional[str] = None
    used_fields: List[str] = []
    suggested_fields: Optional[List[str]] = None
    suggested_statement: str

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = None
    code: Optional[str] = ""
    selects: List[SelectItem] = []

# ===== Helpers =====
def is_obsolete_table(table: str) -> bool:
    return table.upper() in TABLE_MAPPING

def remediation_comment(table: str) -> str:
    return f"* TODO: {table.upper()} is obsolete in S/4HANA (SAP Note 3404390). Use released CDS view {TABLE_MAPPING[table.upper()]['source']} instead. Adjust field mappings accordingly."

def remediate_select(sel_text: str, table: str) -> str:
    new_table = TABLE_MAPPING[table.upper()]["source"]
    comment = remediation_comment(table)
    return f"SELECT * FROM {new_table}\n{comment}"

def remediate_nonselect(stmt_prefix: str, table: str) -> str:
    new_table = TABLE_MAPPING[table.upper()]["source"]
    comment = remediation_comment(table)
    replaced = re.sub(rf"\b{table}\b", new_table, stmt_prefix + table, flags=re.IGNORECASE)
    return f"{replaced}\n{comment}"

# ===== Parse code =====
def parse_and_fill_selects(unit: Unit) -> List[SelectItem]:
    code = unit.code or ""
    findings: List[SelectItem] = []

    # SELECT statements
    for m in SELECT_RE.finditer(code):
        table = m.group("table")
        if is_obsolete_table(table):
            findings.append(SelectItem(
                table=table,
                target_type="itab" if m.group("into_tab") else "wa",
                target_name=(m.group("into_tab") or m.group("into_wa")),
                used_fields=[],
                suggested_fields=None,
                suggested_statement=remediate_select(m.group("full"), table)
            ))

    # Non-SELECT J_1BBRANCH usages (TABLES:, TYPE, LIKE, JOIN etc.)
    for m in GENERIC_USAGE_RE.finditer(code):
        table = m.group("table")
        stmt_prefix = m.group("stmt")
        if is_obsolete_table(table):
            findings.append(SelectItem(
                table=table,
                target_type=None,
                target_name=None,
                used_fields=[],
                suggested_fields=None,
                suggested_statement=remediate_nonselect(stmt_prefix, table)
            ))

    return findings

# ===== Summariser =====
def summarize_selects(unit: Unit) -> Dict[str, Any]:
    tables_count: Dict[str, int] = {}
    flagged = []
    for s in unit.selects:
        tbl_upper = s.table.upper()
        tables_count[tbl_upper] = tables_count.get(tbl_upper, 0) + 1
        flagged.append({"table": s.table, "reason": remediation_comment(s.table)})
    return {
        "program": unit.pgm_name,
        "include": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name,
        "stats": {
            "total_statements": len(unit.selects),
            "tables_frequency": tables_count,
            "note_3404390_flags": flagged
        }
    }

# ===== LLM Prompt Setup =====
SYSTEM_MSG = "You are a precise ABAP reviewer familiar with SAP Note 3404390. Output strict JSON only."

USER_TEMPLATE = """
You are assessing ABAP code usage in light of SAP Note 3404390 (Obsolete business place table in S/4HANA).

From S/4HANA onwards, J_1BBRANCH is obsolete and replaced by the released CDS view P_BusinessPlace.

We provide program/include/unit metadata, and statement analysis.

Your tasks:
1) Produce a concise **assessment** highlighting:
   - Which statements reference J_1BBRANCH.
   - Why migration to P_BusinessPlace is needed.
   - Potential functional and data impact.
2) Produce an **LLM remediation prompt** to:
   - Scan ABAP code in this unit for usage of J_1BBRANCH.
   - Replace SELECT statements with `SELECT * FROM P_BusinessPlace` and add TODO.
   - Replace non-SELECT usages (JOIN, TABLES, TYPE, LIKE) with P_BusinessPlace and add TODO.

Return ONLY strict JSON:
{{
  "assessment": "<concise note 3404390 impact>",
  "llm_prompt": "<prompt for LLM code fixer>"
}}

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {unit_type}
- Unit name: {unit_name}

Analysis:
{plan_json}

selects (JSON):
{selects_json}
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ]
)

llm = ChatOpenAI(model=OPENAI_MODEL)
parser = JsonOutputParser()
chain = prompt | llm | parser

# ===== LLM Call =====
def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    plan = summarize_selects(unit)
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    selects_json = json.dumps([s.model_dump() for s in unit.selects], ensure_ascii=False, indent=2)

    try:
        return chain.invoke({
            "pgm_name": unit.pgm_name,
            "inc_name": unit.inc_name,
            "unit_type": unit.type,
            "unit_name": unit.name,
            "table_list": ", ".join(sorted(TABLE_MAPPING.keys())),
            "plan_json": plan_json,
            "selects_json": selects_json
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

# ===== API POST =====
@app.post("/assess-businessplace-migration")
async def assess_businessplace_migration(units: List[Unit]) -> List[Dict[str, Any]]:
    out = []
    for u in units:
                # Fill selects with regex parser
        u.selects = parse_and_fill_selects(u)
        # Get LLM output
        llm_out = llm_assess_and_prompt(u)
        obj = u.model_dump()
        obj.pop("selects", None)  # remove raw selects from output
        obj["assessment"] = llm_out.get("assessment", "")
        obj["llm_prompt"] = llm_out.get("llm_prompt", "")
        out.append(obj)
    return out


@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}