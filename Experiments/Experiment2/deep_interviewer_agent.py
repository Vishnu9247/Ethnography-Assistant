"""
Deep Interviewer Agent
-----------------------
This agent:
1. Reads a persona's Problem from the Excel file
2. Generates ALL questions needed to fully understand and solve the problem
3. For each question, asks the persona (via ethnographic_pipeline) and evaluates the answer
4. If the answer is insufficient, generates follow-up questions (1.1, 1.2, ...) until
   the context is satisfactory, then moves to the next main question
5. Logs everything to chat_logs/<Name>.txt with numbered labels (1, 2, 1.1, 1.2, etc.)

Usage:
    python deep_interviewer_agent.py                # runs row 0 by default
    python deep_interviewer_agent.py --index 3      # runs row 3
    python deep_interviewer_agent.py --all          # runs all rows
"""

import argparse
from typing import TypedDict

from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from ethnographic_pipeline import (
    load_personas,
    build_graph as build_persona_graph,
    get_output_path,
    EXCEL_PATH,
)

# ─── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "llama3.2"
llm = OllamaLLM(model=MODEL_NAME)


# ─── LangGraph states ──────────────────────────────────────────────────────────

class QuestionGenState(TypedDict):
    problem: str
    questions: list[str]

class EvalState(TypedDict):
    problem: str
    question: str
    answer: str
    is_sufficient: bool
    followup_question: str


# ─── Helper: parse a numbered list from LLM output ─────────────────────────────

def parse_numbered_list(text: str) -> list[str]:
    items = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line and line[0].isdigit():
            item = line.split('.', 1)[-1].split(')', 1)[-1].strip()
            if item:
                items.append(item)
    return items


# ─── Node: generate all questions for the problem ──────────────────────────────

def generate_all_questions(state: QuestionGenState) -> QuestionGenState:
    system_prompt = """You are an expert ethnographic researcher.
Given a problem, generate a comprehensive numbered list of open-ended interview questions
that would fully uncover the root causes, lived experience, emotional impact, past attempts,
and desired outcomes related to the problem.

Return ONLY a numbered list of questions. No preamble or explanation."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Problem: {state['problem']}\n\nGenerate all necessary interview questions."),
    ])

    return {**state, "questions": parse_numbered_list(response)}


# ─── Node: evaluate if the answer is sufficient ────────────────────────────────

def evaluate_answer(state: EvalState) -> EvalState:
    system_prompt = """You are an ethnographic research analyst evaluating interview answers.

Given the overall problem, the question asked, and the answer received, decide:
1. Is the answer sufficient to build meaningful context about this aspect of the problem?
2. If NOT sufficient, write one focused follow-up question to get the missing information.

Respond in exactly this format:
SUFFICIENT: yes
or
SUFFICIENT: no
FOLLOWUP: <your follow-up question here>

Nothing else."""

    user_prompt = (
        f"Problem: {state['problem']}\n\n"
        f"Question asked: {state['question']}\n\n"
        f"Answer received: {state['answer']}"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]).strip()

    is_sufficient = response.lower().startswith("sufficient: yes")
    followup = ""
    if not is_sufficient:
        for line in response.splitlines():
            if line.strip().lower().startswith("followup:"):
                followup = line.split(":", 1)[-1].strip()
                break

    return {**state, "is_sufficient": is_sufficient, "followup_question": followup}


# ─── Build graphs ──────────────────────────────────────────────────────────────

def build_question_gen_graph():
    graph = StateGraph(QuestionGenState)
    graph.add_node("generate_all_questions", generate_all_questions)
    graph.set_entry_point("generate_all_questions")
    graph.add_edge("generate_all_questions", END)
    return graph.compile()

def build_eval_graph():
    graph = StateGraph(EvalState)
    graph.add_node("evaluate_answer", evaluate_answer)
    graph.set_entry_point("evaluate_answer")
    graph.add_edge("evaluate_answer", END)
    return graph.compile()


# ─── Ask the persona a single question via the pipeline ────────────────────────

def ask_persona(persona: dict, question: str, persona_graph) -> str:
    state  = {"persona": persona, "question": question, "answer": ""}
    result = persona_graph.invoke(state)
    return result["answer"]


# ─── Log a Q&A entry to the file ───────────────────────────────────────────────

def log_entry(filepath: str, label: str, question: str, answer: str):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"[{label}] Question: {question}\n")
        f.write(f"      Answer  : {answer}\n")
        f.write("-" * 80 + "\n")


# ─── Main deep interview runner ────────────────────────────────────────────────

def run_deep_interviewer(persona_index: int):
    personas     = load_personas(EXCEL_PATH)
    persona      = personas[persona_index]
    filepath     = get_output_path(persona["Name"])
    persona_graph = build_persona_graph()
    qgen_graph   = build_question_gen_graph()
    eval_graph   = build_eval_graph()

    print(f"\n{'='*60}")
    print(f"Persona : {persona['Name']}")
    print(f"Problem : {persona['Problem']}")
    print(f"{'='*60}\n")

    # Write file header
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Deep Interview with: {persona['Name']}\n")
        f.write("=" * 80 + "\n\n")

    # Step 1: generate all questions
    result    = qgen_graph.invoke({"problem": persona["Problem"], "questions": []})
    questions = result["questions"]

    print(f"Generated {len(questions)} questions.\n")

    # Step 2: for each question, ask → evaluate → follow-up if needed
    for q_num, question in enumerate(questions, start=1):
        main_label = str(q_num)
        print(f"\n[{main_label}] {question}")

        # Ask main question
        answer = ask_persona(persona, question, persona_graph)
        print(f"  A: {answer}")
        log_entry(filepath, main_label, question, answer)

        # Evaluate and follow up if needed
        followup_count = 0
        current_question = question
        current_answer   = answer

        while True:
            eval_result = eval_graph.invoke({
                "problem":   persona["Problem"],
                "question":  current_question,
                "answer":    current_answer,
                "is_sufficient": False,
                "followup_question": "",
            })

            if eval_result["is_sufficient"]:
                print(f"  ✓ Answer sufficient. Moving to next question.")
                break

            followup_count += 1
            followup_label    = f"{q_num}.{followup_count}"
            followup_question = eval_result["followup_question"]

            if not followup_question:
                # Safety: if no follow-up was parsed, move on
                break

            print(f"\n  [{followup_label}] Follow-up: {followup_question}")
            followup_answer = ask_persona(persona, followup_question, persona_graph)
            print(f"  A: {followup_answer}")
            log_entry(filepath, followup_label, followup_question, followup_answer)

            # Next evaluation uses the follow-up Q&A as context
            current_question = followup_question
            current_answer   = followup_answer

    print(f"\nInterview complete. Saved to: {filepath}")


# ─── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0,
                        help="Row index (0-based) of the persona to interview")
    parser.add_argument("--all", action="store_true",
                        help="Run deep interviews for all personas in the Excel file")
    args = parser.parse_args()

    if args.all:
        personas = load_personas(EXCEL_PATH)
        for i in range(len(personas)):
            run_deep_interviewer(persona_index=i)
    else:
        run_deep_interviewer(persona_index=args.index)
