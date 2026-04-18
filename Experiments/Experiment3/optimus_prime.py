"""
optimus_prime.py — Problem Definition Agent
Conducts a structured intake conversation, then writes an intake document.
"""

import re
import os
from datetime import datetime
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")

# ── State ──────────────────────────────────────────────────────────────────────

class IntakeState(TypedDict):
    messages: list[dict]          # {role, content}
    user_profile: dict            # name, age, occupation, context
    problem_statement: str
    fields_to_explore: list[dict] # [{field, reason, seed_questions:[]}]
    phase: str                    # "greeting" | "problem" | "clarify" | "fields" | "done"
    output_path: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def chat(state: IntakeState, system: str, user_msg: str) -> str:
    history = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in state["messages"][-10:]
    )
    prompt = f"{system}\n\nConversation so far:\n{history}\n\nUSER: {user_msg}\nASSISTANT:"
    response = llm.invoke(prompt)
    return response.content.strip()


def ask_user(prompt: str) -> str:
    print(f"\nOptimus Prime: {prompt}")
    return input("You: ").strip()


def evaluate_problem_clarity(problem_text: str) -> bool:
    prompt = (
        "You are evaluating whether a problem statement is clear, specific, and actionable.\n"
        f"Problem: {problem_text}\n"
        "Reply with only YES if the problem is clear and specific, or NO if it needs clarification."
    )
    result = llm.invoke(prompt).content.strip().upper()
    return result.startswith("YES")


# ── Nodes ─────────────────────────────────────────────────────────────────────

def greet_node(state: IntakeState) -> IntakeState:
    greeting = (
        "Hello! I'm Optimus Prime, your ethnographic research assistant. "
        "I'm here to help you define your problem clearly so we can explore the right areas of your life.\n\n"
        "Let's start with a brief introduction. What's your name, age, occupation, and what brings you here today?"
    )
    answer = ask_user(greeting)
    state["messages"].append({"role": "user", "content": answer})

    # Parse profile with LLM
    parse_prompt = (
        "Extract name, age, occupation, and any personal context from this text. "
        "Reply as: NAME: ...\nAGE: ...\nOCCUPATION: ...\nCONTEXT: ...\n"
        f"Text: {answer}"
    )
    parsed = llm.invoke(parse_prompt).content.strip()
    profile = {}
    for line in parsed.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            profile[key.strip().lower()] = val.strip()

    state["user_profile"] = profile
    state["phase"] = "problem"
    return state


def problem_node(state: IntakeState) -> IntakeState:
    name = state["user_profile"].get("name", "there")
    first_name = name.split()[0] if name else "there"
    question = (
        f"Thanks, {first_name}! Now, in your own words, please describe the problem or challenge "
        "you're facing that you'd like to better understand."
    )
    answer = ask_user(question)
    state["messages"].append({"role": "user", "content": answer})
    state["problem_statement"] = answer
    state["phase"] = "clarify"
    return state


def clarify_node(state: IntakeState) -> IntakeState:
    is_clear = evaluate_problem_clarity(state["problem_statement"])
    if is_clear:
        state["phase"] = "fields"
        return state

    # Generate a targeted follow-up
    system = (
        "You are Optimus Prime, an ethnographic research intake agent. "
        "Your goal is to help the user articulate their problem clearly. "
        "Ask ONE focused follow-up question to make the problem more specific and actionable."
    )
    follow_up = chat(state, system, state["problem_statement"])
    answer = ask_user(follow_up)

    state["messages"].append({"role": "assistant", "content": follow_up})
    state["messages"].append({"role": "user", "content": answer})
    state["problem_statement"] += " " + answer

    # Check again after one round — loop handled by graph routing
    state["phase"] = "clarify_check"
    return state


def clarify_check_node(state: IntakeState) -> IntakeState:
    is_clear = evaluate_problem_clarity(state["problem_statement"])
    state["phase"] = "fields" if is_clear else "clarify"
    return state


def fields_node(state: IntakeState) -> IntakeState:
    problem = state["problem_statement"]
    profile = state["user_profile"]

    prompt = (
        f"You are an expert ethnographic researcher. A user has described the following problem:\n\n"
        f"Problem: {problem}\n"
        f"User Profile: {profile}\n\n"
        "Identify 4–6 fields of the person's life that are most relevant to explore ethnographically "
        "(e.g., daily routine, relationships, work environment, emotional patterns, finances, health habits).\n\n"
        "For each field, provide:\n"
        "1. Field name\n"
        "2. Why it's relevant to this problem (1–2 sentences)\n"
        "3. 3 seed questions to ask\n\n"
        "Format each field as:\n"
        "FIELD: <name>\n"
        "REASON: <reason>\n"
        "Q1: <question>\n"
        "Q2: <question>\n"
        "Q3: <question>\n"
        "---\n"
    )
    response = llm.invoke(prompt).content.strip()

    # Parse fields
    fields = []
    blocks = response.split("---")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        field_data = {}
        questions = []
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("FIELD:"):
                field_data["field"] = line.replace("FIELD:", "").strip()
            elif line.startswith("REASON:"):
                field_data["reason"] = line.replace("REASON:", "").strip()
            elif re.match(r"^Q\d+:", line):
                questions.append(re.sub(r"^Q\d+:", "", line).strip())
        if field_data.get("field"):
            field_data["seed_questions"] = questions
            fields.append(field_data)

    state["fields_to_explore"] = fields
    state["phase"] = "done"
    return state


def write_document_node(state: IntakeState) -> IntakeState:
    profile = state["user_profile"]
    name = profile.get("name", "Unknown User")
    parts = name.split()
    first = parts[0] if parts else "Unknown"
    last = parts[-1] if len(parts) > 1 else "User"

    now = datetime.now()
    filename = f"{first}_{last}-{now.strftime('%m%d')}-{now.strftime('%H%M')}.txt"
    os.makedirs("intake_docs", exist_ok=True)
    filepath = os.path.join("intake_docs", filename)

    # Generate "why this problem matters" with LLM
    why_prompt = (
        f"In 2–3 sentences, explain why the following problem matters for this person's life "
        f"and why it's worth investigating ethnographically:\n\nProblem: {state['problem_statement']}\n"
        f"Profile: {profile}"
    )
    why_matters = llm.invoke(why_prompt).content.strip()

    lines = [
        "=" * 70,
        "ETHNOGRAPHIC INTAKE DOCUMENT",
        "=" * 70,
        "",
        "USER PROFILE",
        "-" * 40,
        f"Name:       {profile.get('name', 'N/A')}",
        f"Age:        {profile.get('age', 'N/A')}",
        f"Occupation: {profile.get('occupation', 'N/A')}",
        f"Context:    {profile.get('context', 'N/A')}",
        "",
        "PROBLEM STATEMENT",
        "-" * 40,
        state["problem_statement"],
        "",
        "WHY THIS PROBLEM MATTERS",
        "-" * 40,
        why_matters,
        "",
        "FIELDS TO EXPLORE",
        "-" * 40,
    ]

    for i, field in enumerate(state["fields_to_explore"], 1):
        lines.append(f"\n{i}. {field.get('field', 'Unnamed Field')}")
        lines.append(f"   Relevance: {field.get('reason', '')}")
        lines.append("   Seed Questions:")
        for q in field.get("seed_questions", []):
            lines.append(f"     - {q}")

    lines += [
        "",
        "=" * 70,
        f"Generated: {now.strftime('%Y-%m-%d %H:%M')}",
        "=" * 70,
    ]

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    state["output_path"] = filepath
    print(f"\n[Optimus Prime] Intake document saved to: {filepath}")
    return state


# ── Graph ─────────────────────────────────────────────────────────────────────

def route_after_clarify(state: IntakeState) -> str:
    return state["phase"]


def build_graph() -> StateGraph:
    graph = StateGraph(IntakeState)

    graph.add_node("greet", greet_node)
    graph.add_node("problem", problem_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("clarify_check", clarify_check_node)
    graph.add_node("fields", fields_node)
    graph.add_node("write_document", write_document_node)

    graph.set_entry_point("greet")
    graph.add_edge("greet", "problem")
    graph.add_edge("problem", "clarify")
    graph.add_conditional_edges(
        "clarify",
        route_after_clarify,
        {
            "fields": "fields",
            "clarify_check": "clarify_check",
        }
    )
    graph.add_conditional_edges(
        "clarify_check",
        route_after_clarify,
        {
            "fields": "fields",
            "clarify": "clarify",
        }
    )
    graph.add_edge("fields", "write_document")
    graph.add_edge("write_document", END)

    return graph.compile()


def run() -> str:
    """Run Optimus Prime and return the path to the intake document."""
    app = build_graph()
    initial_state: IntakeState = {
        "messages": [],
        "user_profile": {},
        "problem_statement": "",
        "fields_to_explore": [],
        "phase": "greeting",
        "output_path": "",
    }
    final_state = app.invoke(initial_state)
    return final_state["output_path"]


if __name__ == "__main__":
    path = run()
    print(f"\nDone. Intake document: {path}")
