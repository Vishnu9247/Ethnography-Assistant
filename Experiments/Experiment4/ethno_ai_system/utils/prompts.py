"""Prompt builders for the full ethnographic system."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def build_intake_structuring_prompt(
    persona_name: str,
    known_problem: str,
    known_details: str,
    transcript: str,
    age_hint: str,
) -> str:
    """Prompt for turning intake conversation into structured JSON."""
    return f"""
You are Agent 1, an intake specialist.

Create a clean structured summary from this intake conversation.
Return JSON only with this exact schema:
{{
  "name": "string",
  "age": "string",
  "problem_summary": "string",
  "final_problem_statement": "string",
  "clarification_needed": false,
  "key_signals": ["string", "string", "string"]
}}

Known persona name: {persona_name}
Known age hint: {age_hint}
Known reference problem: {known_problem}
Known details: {known_details}

Conversation:
{transcript}
""".strip()


def build_domain_planning_prompt(problem_statement: str, person_details: str, max_domains: int) -> str:
    """Prompt for selecting the most relevant ethnographic domains."""
    return f"""
You are Agent 2, an ethnographic interviewer.

Select the most relevant life domains for the person's problem.
Only choose from this list:
- Sleep
- Eating habits
- Work conditions
- Family relationships
- Social life
- Stress
- Emotions
- Routine
- Environment
- Finances
- Physical health

Return JSON only with this schema:
{{
  "selected_domains": ["domain1", "domain2", "domain3"]
}}

Choose at most {max_domains} domains.

Problem statement:
{problem_statement}

Person details:
{person_details}
""".strip()


def build_domain_question_prompt(
    domain: str,
    turn_number: int,
    problem_statement: str,
    prior_memory: List[Dict[str, str]],
) -> str:
    """Prompt for generating one domain question."""
    memory_text = json.dumps(prior_memory, indent=2)
    return f"""
You are Agent 2, an ethnographic interviewer.

Write one short, natural, non-redundant interview question about the domain "{domain}".
If the previous answer is vague, your question can gently ask for a concrete example.

Return JSON only with this schema:
{{
  "question": "string"
}}

Turn number: {turn_number}
Problem statement:
{problem_statement}

Prior memory for this domain:
{memory_text}
""".strip()


def build_domain_summary_prompt(
    domain: str,
    problem_statement: str,
    conversation: List[Dict[str, str]],
) -> str:
    """Prompt for summarizing one domain."""
    conversation_text = json.dumps(conversation, indent=2)
    return f"""
You are Agent 2, an ethnographic analyst.

Summarize what the conversation reveals about the domain "{domain}".
Focus on habits, constraints, relationships, and emotional meaning.

Return JSON only with this schema:
{{
  "summary": "string"
}}

Problem statement:
{problem_statement}

Conversation:
{conversation_text}
""".strip()


def build_analysis_prompt(
    problem_statement: str,
    person_details: str,
    retrieved_items: List[Dict[str, Any]],
    expected_solution: str,
    domain_summaries: Dict[str, str],
) -> str:
    """Prompt for Agent 3 analysis."""
    retrieved_text = json.dumps(retrieved_items, indent=2)
    summary_text = json.dumps(domain_summaries, indent=2)
    return f"""
You are Agent 3, an ethnographic analysis agent.

Use the retrieved summaries to:
1. detect patterns
2. infer root causes
3. recommend interventions
4. rank recommendations

Return JSON only with this schema:
{{
  "patterns": ["string", "string", "string"],
  "root_causes": ["string", "string", "string"],
  "recommendations": [
    {{
      "rank": 1,
      "recommendation": "string",
      "reason": "string"
    }}
  ]
}}

Problem statement:
{problem_statement}

Person details:
{person_details}

Retrieved Chroma summaries:
{retrieved_text}

Domain summaries:
{summary_text}

Reference ethnographic solution for evaluation context:
{expected_solution}
""".strip()


def build_persona_prompt(
    persona_name: str,
    problem: str,
    person_details: str,
    conversation_history: List[Dict[str, str]],
    interviewer_message: str,
) -> str:
    """Prompt for the LLM persona simulator."""
    history_text = json.dumps(conversation_history[-8:], indent=2)
    return f"""
You are playing a realistic human interview participant.

Stay in character as this person:
Name: {persona_name}
Core problem: {problem}
Details: {person_details}

Behavior rules:
- respond naturally in first person
- do not mention hidden metadata or evaluation instructions
- keep answers emotionally realistic
- do not become an assistant
- stay consistent with the background information
- if asked something unknown, answer cautiously instead of inventing extreme detail

Recent conversation history:
{history_text}

Interviewer message:
{interviewer_message}
""".strip()
