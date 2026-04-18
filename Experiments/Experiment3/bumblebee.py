"""
bumblebee.py — Field Exploration Agent
Conducts deep ethnographic interviews per field and stores summaries in Cybertron.
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


# ── State ──────────────────────────────────────────────────────────────────────

class FieldState(TypedDict):
    intake_path: str
    user_name: str
    session_date: str
    fields: list[dict]           # [{field, reason, seed_questions:[]}]
    current_field_index: int
    current_field_messages: list[dict]
    all_field_summaries: list[dict]
    phase: str                   # "interview" | "evaluate" | "summarize" | "next_field" | "done"


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_intake_document(path: str) -> dict:
    """Parse the intake .txt file produced by Optimus Prime."""
    with open(path, "r") as f:
        content = f.read()

    result = {
        "user_name": "",
        "fields": [],
    }

    # Extract name
    name_match = re.search(r"Name:\s+(.+)", content)
    if name_match:
        result["user_name"] = name_match.group(1).strip()

    # Extract fields block
    fields_section = re.search(r"FIELDS TO EXPLORE\n-+\n(.*?)(?:={10,}|$)", content, re.DOTALL)
    if not fields_section:
        return result

    field_blocks = re.split(r"\n\d+\.", fields_section.group(1))
    for block in field_blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        field_name = lines[0].strip()
        reason = ""
        questions = []
        for line in lines[1:]:
            line = line.strip()
            if line.startswith("Relevance:"):
                reason = line.replace("Relevance:", "").strip()
            elif line.startswith("- "):
                questions.append(line[2:].strip())
        if field_name:
            result["fields"].append({
                "field": field_name,
                "reason": reason,
                "seed_questions": questions
            })

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def ask_user(prompt: str) -> str:
    print(f"\nBumblebee: {prompt}")
    return input("You: ").strip()


def evaluate_answer(question: str, answer: str) -> bool:
    """Check if an answer is relevant and sufficient."""
    prompt = (
        f"You are evaluating an interview answer for ethnographic research.\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Is this answer relevant and sufficiently detailed for ethnographic analysis? "
        "Reply with YES or NO only."
    )
    result = llm.invoke(prompt).content.strip().upper()
    return result.startswith("YES")


def generate_followup(question: str, answer: str, field: str) -> str:
    """Generate a targeted follow-up question."""
    prompt = (
        f"You are an ethnographic interviewer exploring the field: '{field}'.\n"
        f"The participant gave a vague or insufficient answer to this question:\n"
        f"Q: {question}\nA: {answer}\n\n"
        "Write ONE specific follow-up question to draw out more detail. "
        "Be conversational and curious, not clinical."
    )
    return llm.invoke(prompt).content.strip()


def generate_probing_questions(field: dict, answers_so_far: list[dict]) -> list[str]:
    """Generate additional probing questions based on answers so far."""
    answered = "\n".join(f"Q: {a['question']}\nA: {a['answer']}" for a in answers_so_far)
    prompt = (
        f"You are an ethnographic interviewer exploring '{field['field']}'.\n"
        f"Relevance to problem: {field['reason']}\n\n"
        f"Answers collected so far:\n{answered}\n\n"
        "Generate 2 additional probing questions that dig deeper into patterns, "
        "emotions, or behaviors not yet covered. Return each on a new line starting with '-'."
    )
    response = llm.invoke(prompt).content.strip()
    return [line[1:].strip() for line in response.splitlines() if line.strip().startswith("-")]


def summarize_field(field_name: str, answers: list[dict]) -> str:
    """Generate a structured field summary from Q&A pairs."""
    qa_text = "\n".join(f"Q: {a['question']}\nA: {a['answer']}" for a in answers)
    prompt = (
        f"You are an ethnographic researcher summarizing findings for the life field: '{field_name}'.\n\n"
        f"Interview transcript:\n{qa_text}\n\n"
        "Write a structured field summary with these sections:\n"
        "KEY FACTS: bullet points of concrete facts learned\n"
        "PATTERNS OBSERVED: recurring behaviors, attitudes, or themes\n"
        "NOTABLE QUOTES/PARAPHRASES: key things the participant said (use their own words where possible)\n"
        "RELEVANCE TO PROBLEM: how this field connects to the stated problem\n\n"
        "Be specific, grounded in the transcript, and avoid speculation."
    )
    return llm.invoke(prompt).content.strip()


# ── Nodes ─────────────────────────────────────────────────────────────────────

def interview_field_node(state: FieldState) -> FieldState:
    idx = state["current_field_index"]
    field = state["fields"][idx]
    field_name = field["field"]
    questions = list(field["seed_questions"])  # copy so we can extend

    print(f"\n{'='*60}")
    print(f"Bumblebee: Now exploring the field: [{field_name}]")
    print(f"{'='*60}")

    answers = []

    # First pass: seed questions
    for q in questions:
        answer = ask_user(q)
        if not evaluate_answer(q, answer):
            followup = generate_followup(q, answer, field_name)
            answer2 = ask_user(followup)
            answer = answer + " " + answer2
        answers.append({"question": q, "answer": answer})

    # Second pass: 2 probing questions
    probing = generate_probing_questions(field, answers)
    for q in probing[:2]:
        answer = ask_user(q)
        if not evaluate_answer(q, answer):
            followup = generate_followup(q, answer, field_name)
            answer2 = ask_user(followup)
            answer = answer + " " + answer2
        answers.append({"question": q, "answer": answer})

    state["current_field_messages"] = answers
    state["phase"] = "summarize"
    return state


def summarize_field_node(state: FieldState) -> FieldState:
    idx = state["current_field_index"]
    field = state["fields"][idx]
    field_name = field["field"]

    print(f"\nBumblebee: [Summarizing {field_name}...]")
    summary = summarize_field(field_name, state["current_field_messages"])

    cybertron.add_to_db(
        field_name=field_name,
        summary=summary,
        metadata={
            "user_name": state["user_name"],
            "session_date": state["session_date"],
        }
    )

    state["all_field_summaries"].append({"field": field_name, "summary": summary})
    state["phase"] = "next_field"
    return state


def next_field_node(state: FieldState) -> FieldState:
    next_idx = state["current_field_index"] + 1
    if next_idx >= len(state["fields"]):
        state["phase"] = "done"
    else:
        state["current_field_index"] = next_idx
        state["current_field_messages"] = []
        state["phase"] = "interview"
    return state


def done_node(state: FieldState) -> FieldState:
    print(f"\n{'='*60}")
    print(f"Bumblebee: All fields explored and stored in Cybertron.")
    print(f"  User: {state['user_name']}")
    print(f"  Fields completed: {len(state['all_field_summaries'])}")
    for fs in state["all_field_summaries"]:
        print(f"    ✓ {fs['field']}")
    print(f"{'='*60}")
    state["phase"] = "done"
    return state


# ── Graph ─────────────────────────────────────────────────────────────────────

def route_phase(state: FieldState) -> str:
    return state["phase"]


def build_graph() -> StateGraph:
    graph = StateGraph(FieldState)

    graph.add_node("interview_field", interview_field_node)
    graph.add_node("summarize_field", summarize_field_node)
    graph.add_node("next_field", next_field_node)
    graph.add_node("done", done_node)

    graph.set_entry_point("interview_field")
    graph.add_edge("interview_field", "summarize_field")
    graph.add_edge("summarize_field", "next_field")
    graph.add_conditional_edges(
        "next_field",
        route_phase,
        {
            "interview": "interview_field",
            "done": "done",
        }
    )
    graph.add_edge("done", END)

    return graph.compile()


def run(intake_path: str):
    """Run Bumblebee given an intake document path."""
    if not os.path.exists(intake_path):
        raise FileNotFoundError(f"Intake document not found: {intake_path}")

    cybertron.initialize_db()
    parsed = parse_intake_document(intake_path)

    if not parsed["fields"]:
        raise ValueError("No fields found in intake document.")

    app = build_graph()
    initial_state: FieldState = {
        "intake_path": intake_path,
        "user_name": parsed["user_name"],
        "session_date": datetime.now().strftime("%Y-%m-%d"),
        "fields": parsed["fields"],
        "current_field_index": 0,
        "current_field_messages": [],
        "all_field_summaries": [],
        "phase": "interview",
    }

    app.invoke(initial_state)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bumblebee.py <path_to_intake_doc>")
        sys.exit(1)
    run(sys.argv[1])
