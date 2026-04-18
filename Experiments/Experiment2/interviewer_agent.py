import argparse
from typing import TypedDict

from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from ethnographic_pipeline import load_personas, run_pipeline, EXCEL_PATH


MODEL_NAME = "llama3.2"
llm = OllamaLLM(model=MODEL_NAME)



class QuestionGenState(TypedDict):
    problem: str
    questions: list[str]



def generate_questions(state: QuestionGenState) -> QuestionGenState:
    system_prompt = """You are an expert ethnographic researcher designing an interview.
Your goal is to deeply understand a person's problem and help identify ways to solve it.

Given a problem description, generate exactly 4 to 5 open-ended interview questions.
These questions should:
- Help uncover the root cause of the problem
- Explore the person's emotions and daily experience
- Identify what they have already tried
- Understand what support or solution they are looking for

Return ONLY a numbered list of questions, nothing else. No preamble, no explanation."""

    user_prompt = f"Problem: {state['problem']}\n\nGenerate 4-5 interview questions."

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    # Parse numbered list into a clean list of strings
    questions = []
    for line in response.strip().splitlines():
        line = line.strip()
        if line and line[0].isdigit():
            # Strip leading "1. " / "1) " etc.
            question = line.split('.', 1)[-1].split(')', 1)[-1].strip()
            if question:
                questions.append(question)

    return {**state, "questions": questions}


def build_question_graph():
    graph = StateGraph(QuestionGenState)
    graph.add_node("generate_questions", generate_questions)
    graph.set_entry_point("generate_questions")
    graph.add_edge("generate_questions", END)
    return graph.compile()


def run_interviewer(persona_index: int):
    personas = load_personas(EXCEL_PATH)
    persona  = personas[persona_index]

    print(f"\n{'='*60}")
    print(f"Persona  : {persona['Name']}")
    print(f"Problem  : {persona['Problem']}")
    print(f"{'='*60}\n")

    # Step 1: generate questions
    graph  = build_question_graph()
    result = graph.invoke({"problem": persona["Problem"], "questions": []})
    questions = result["questions"]

    print("Generated questions:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")
    print()

    # Step 2: feed questions into the pipeline
    run_pipeline(questions=questions, persona_index=persona_index)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0,
                        help="Row index (0-based) of the persona to interview")
    parser.add_argument("--all", action="store_true",
                        help="Run interviews for all personas in the Excel file")
    args = parser.parse_args()

    if args.all:
        personas = load_personas(EXCEL_PATH)
        for i in range(len(personas)):
            run_interviewer(persona_index=i)
    else:
        run_interviewer(persona_index=args.index)
