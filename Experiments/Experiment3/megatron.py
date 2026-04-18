"""
megatron.py — Synthesis & Solution Agent
Retrieves field summaries from Cybertron and produces a grounded solution report.
"""

import re
import sys
import os
from datetime import datetime
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

import cybertron

llm = ChatOllama(model="llama3.2")

MAX_RETRIEVAL_ROUNDS = 3


# ── State ──────────────────────────────────────────────────────────────────────

class SynthesisState(TypedDict):
    intake_path: str
    user_name: str
    problem_statement: str
    fields_listed: list[str]
    retrieval_queries: list[str]
    retrieved_summaries: list[dict]   # [{field, content, metadata}]
    retrieval_round: int
    synthesis_sufficient: bool
    report_text: str
    output_path: str
    phase: str  # "plan" | "retrieve" | "evaluate" | "synthesize" | "write" | "done"


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_intake_document(path: str) -> dict:
    with open(path, "r") as f:
        content = f.read()

    result = {"user_name": "", "problem_statement": "", "fields": []}

    name_match = re.search(r"Name:\s+(.+)", content)
    if name_match:
        result["user_name"] = name_match.group(1).strip()

    prob_match = re.search(r"PROBLEM STATEMENT\n-+\n(.+?)(?:\n\n|\Z)", content, re.DOTALL)
    if prob_match:
        result["problem_statement"] = prob_match.group(1).strip()

    fields_section = re.search(r"FIELDS TO EXPLORE\n-+\n(.*?)(?:={10,}|$)", content, re.DOTALL)
    if fields_section:
        field_lines = re.findall(r"\d+\.\s+(.+)", fields_section.group(1))
        result["fields"] = [f.strip() for f in field_lines]

    return result


# ── Nodes ─────────────────────────────────────────────────────────────────────

def plan_node(state: SynthesisState) -> SynthesisState:
    """Generate targeted retrieval queries from the problem and field list."""
    prompt = (
        f"You are Megatron, an analytical synthesis agent.\n"
        f"Problem: {state['problem_statement']}\n"
        f"Fields explored: {', '.join(state['fields_listed'])}\n\n"
        "Generate 5–7 specific retrieval queries to find the most relevant ethnographic findings "
        "in the database. Each query should target a different angle of the problem.\n"
        "Return one query per line, starting with '-'."
    )
    response = llm.invoke(prompt).content.strip()
    queries = [line[1:].strip() for line in response.splitlines() if line.strip().startswith("-")]
    state["retrieval_queries"] = queries
    state["phase"] = "retrieve"
    print(f"[Megatron] Planned {len(queries)} retrieval queries.")
    return state


def retrieve_node(state: SynthesisState) -> SynthesisState:
    """Run all queries against Cybertron and collect unique results."""
    seen_ids = {r["id"] for r in state["retrieved_summaries"]}
    new_results = []

    for query in state["retrieval_queries"]:
        results = cybertron.query_db(query, n_results=3)
        for r in results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                new_results.append(r)

    state["retrieved_summaries"] = state["retrieved_summaries"] + new_results
    state["phase"] = "evaluate"
    print(f"[Megatron] Retrieved {len(state['retrieved_summaries'])} unique summaries total.")
    return state


def evaluate_node(state: SynthesisState) -> SynthesisState:
    """Check if retrieved content is sufficient to synthesize a solution."""
    if state["retrieval_round"] >= MAX_RETRIEVAL_ROUNDS:
        state["synthesis_sufficient"] = True
        state["phase"] = "synthesize"
        return state

    summaries_text = "\n\n".join(
        f"[{r['metadata'].get('field_name', '?')}]\n{r['content']}"
        for r in state["retrieved_summaries"]
    )

    prompt = (
        f"Problem: {state['problem_statement']}\n\n"
        f"Retrieved ethnographic summaries:\n{summaries_text}\n\n"
        "Do these summaries provide sufficient evidence to produce a well-grounded solution report? "
        "Reply YES if sufficient, or NO followed by 3 additional queries to try (one per line starting with '-')."
    )
    response = llm.invoke(prompt).content.strip()

    if response.upper().startswith("YES"):
        state["synthesis_sufficient"] = True
        state["phase"] = "synthesize"
    else:
        extra_queries = [line[1:].strip() for line in response.splitlines() if line.strip().startswith("-")]
        state["retrieval_queries"] = extra_queries
        state["retrieval_round"] += 1
        state["phase"] = "retrieve"
        print(f"[Megatron] Retrieval round {state['retrieval_round']}: fetching {len(extra_queries)} more queries.")

    return state


def synthesize_node(state: SynthesisState) -> SynthesisState:
    """Generate the full solution report."""
    summaries_text = "\n\n".join(
        f"[Field: {r['metadata'].get('field_name', '?')}]\n{r['content']}"
        for r in state["retrieved_summaries"]
    )

    prompt = (
        f"You are Megatron, an analytical synthesis agent. "
        f"Your task is to produce a grounded, insight-driven solution report.\n\n"
        f"PROBLEM: {state['problem_statement']}\n\n"
        f"ETHNOGRAPHIC FINDINGS:\n{summaries_text}\n\n"
        "Write a structured report with these exact sections:\n\n"
        "RESTATEMENT OF THE PROBLEM\n"
        "(Synthesize the problem clearly in your own words)\n\n"
        "KEY FINDINGS PER FIELD\n"
        "(For each field, cite specific evidence from the summaries above)\n\n"
        "CROSS-FIELD PATTERNS & INSIGHTS\n"
        "(Identify patterns that span multiple fields)\n\n"
        "ROOT CAUSE ANALYSIS\n"
        "(What underlying factors drive this problem based on the evidence?)\n\n"
        "RECOMMENDED ACTIONS\n"
        "(Practical, prioritized steps the person can take)\n\n"
        "LIMITATIONS & WHAT REMAINS UNKNOWN\n"
        "(Be explicit about gaps or unanswered questions)\n\n"
        "IMPORTANT: Every claim must be traceable to the retrieved summaries above. "
        "Do not fabricate insights. If evidence is insufficient, say so explicitly."
    )

    print("[Megatron] Synthesizing report...")
    report = llm.invoke(prompt).content.strip()
    state["report_text"] = report
    state["phase"] = "write"
    return state


def write_report_node(state: SynthesisState) -> SynthesisState:
    """Write the report to a .txt file."""
    name = state["user_name"]
    parts = name.split()
    first = parts[0] if parts else "Unknown"
    last = parts[-1] if len(parts) > 1 else "User"

    now = datetime.now()
    filename = f"{first}_{last}-report-{now.strftime('%m%d')}-{now.strftime('%H%M')}.txt"
    os.makedirs("reports", exist_ok=True)
    filepath = os.path.join("reports", filename)

    header = "\n".join([
        "=" * 70,
        "MEGATRON ETHNOGRAPHIC SYNTHESIS REPORT",
        "=" * 70,
        f"User: {state['user_name']}",
        f"Generated: {now.strftime('%Y-%m-%d %H:%M')}",
        f"Fields retrieved: {len(set(r['metadata'].get('field_name','?') for r in state['retrieved_summaries']))}",
        f"Retrieval rounds: {state['retrieval_round'] + 1}",
        "=" * 70,
        "",
    ])

    with open(filepath, "w") as f:
        f.write(header + state["report_text"])

    state["output_path"] = filepath
    print(f"[Megatron] Report saved to: {filepath}")
    state["phase"] = "done"
    return state


# ── Graph ─────────────────────────────────────────────────────────────────────

def route_phase(state: SynthesisState) -> str:
    return state["phase"]


def build_graph() -> StateGraph:
    graph = StateGraph(SynthesisState)

    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("write_report", write_report_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        route_phase,
        {
            "retrieve": "retrieve",
            "synthesize": "synthesize",
        }
    )
    graph.add_edge("synthesize", "write_report")
    graph.add_edge("write_report", END)

    return graph.compile()


def run(intake_path: str) -> str:
    """Run Megatron given an intake document path. Returns report path."""
    if not os.path.exists(intake_path):
        raise FileNotFoundError(f"Intake document not found: {intake_path}")

    cybertron.initialize_db()
    parsed = parse_intake_document(intake_path)

    app = build_graph()
    initial_state: SynthesisState = {
        "intake_path": intake_path,
        "user_name": parsed["user_name"],
        "problem_statement": parsed["problem_statement"],
        "fields_listed": parsed["fields"],
        "retrieval_queries": [],
        "retrieved_summaries": [],
        "retrieval_round": 0,
        "synthesis_sufficient": False,
        "report_text": "",
        "output_path": "",
        "phase": "plan",
    }

    final_state = app.invoke(initial_state)
    return final_state["output_path"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python megatron.py <path_to_intake_doc>")
        sys.exit(1)
    path = run(sys.argv[1])
    print(f"\nDone. Report: {path}")
