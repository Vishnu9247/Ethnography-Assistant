import os
import re
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict


EXCEL_PATH   = "ethnographic_training_data_100.xlsx"  # Update path if needed
CHAT_LOGS_DIR = "chat_logs"
MODEL_NAME   = "llama3.2"


def load_personas(path: str) -> list[dict]:
    df = pd.read_excel(path, usecols=["Name", "Problem", "Person Details"])
    return df.to_dict(orient="records")



def build_system_prompt(persona: dict) -> str:
    return f"""You are roleplaying as a real person being interviewed for an ethnographic study.

Name: {persona['Name']}
Your Problem: {persona['Problem']}
About You: {persona['Person Details']}

Stay in character at all times. Respond naturally as this person would — use first person,
reflect their emotions, background, and situation. Do not break character or mention
that you are an AI. Keep answers conversational and authentic."""



class InterviewState(TypedDict):
    persona: dict
    question: str
    answer: str



llm = OllamaLLM(model=MODEL_NAME)

def answer_question(state: InterviewState) -> InterviewState:
    system_prompt = build_system_prompt(state["persona"])
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["question"]),
    ]
    response = llm.invoke(messages)
    return {**state, "answer": response}



def build_graph():
    graph = StateGraph(InterviewState)
    graph.add_node("answer_question", answer_question)
    graph.set_entry_point("answer_question")
    graph.add_edge("answer_question", END)
    return graph.compile()


def get_output_path(persona_name: str) -> str:
    os.makedirs(CHAT_LOGS_DIR, exist_ok=True)
    safe_name = re.sub(r'[^\w\s-]', '', persona_name).strip().replace(' ', '_')
    return os.path.join(CHAT_LOGS_DIR, f"{safe_name}.txt")



def save_to_file(persona_name: str, question: str, answer: str, filepath: str):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"Question: {question}\n")
        f.write(f"Answer  : {answer}\n")
        f.write("-" * 80 + "\n")



def run_pipeline(questions: list[str], persona_index: int = 0):
    """
    Args:
        questions     : List of questions from your architecture agent.
        persona_index : Which row (0-based) from the Excel to use as the persona.
                        Change this per interview session.
    """
    personas  = load_personas(EXCEL_PATH)
    persona   = personas[persona_index]
    graph     = build_graph()
    filepath  = get_output_path(persona["Name"])

    print(f"\n=== Interviewing: {persona['Name']} ===\n")

    # Create/overwrite the file with a header
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Interview with: {persona['Name']}\n")
        f.write("=" * 80 + "\n\n")

    for question in questions:
        print(f"Q: {question}")
        state  = {"persona": persona, "question": question, "answer": ""}
        result = graph.invoke(state)
        answer = result["answer"]
        print(f"A: {answer}\n")
        save_to_file(persona["Name"], question, answer, filepath)

    print(f"Responses saved to: {filepath}")



if __name__ == "__main__":
    # Replace these with questions from your architecture agent.
    # You can also load them from a file or pipe them in from your agent.
    sample_questions = [
        "Can you tell me a bit about yourself and your daily life?",
        "When did you first notice this problem starting to affect you?",
        "How does this situation make you feel on a typical day?",
        "Have you tried talking to anyone about this? What happened?",
        "What would your ideal situation look like?",
    ]

    # persona_index=0 uses the first row (Maria Santos).
    # Change the index to interview a different persona.
    run_pipeline(questions=sample_questions, persona_index=0)