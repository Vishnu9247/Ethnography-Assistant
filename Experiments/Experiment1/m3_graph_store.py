"""
m3_graph_store.py
─────────────────
EXPERIMENT M3 — Graph Store (NetworkX)

Instead of embedding raw text, this experiment extracts structured entities
(people, emotions, topics, events) from each turn and stores them as nodes
in a graph. Edges connect related entities (e.g. "father" — EVOKES — "guilt").

Before each reply, the graph is queried for entities mentioned in the current
query, and their connected neighbours are injected as structured context.

What this measures:
  - Whether structured entity/relationship context beats raw text retrieval
  - Cross-turn linking: can the model connect "mother" in turn 1 with "caregiver
    burden" in turn 4?
  - Richer follow-ups: does the model reference specific named relationships?

Install:
    pip install networkx spacy
    python -m spacy download en_core_web_sm

Run:
    python m3_graph_store.py
"""

import time
import spacy
import networkx as nx
from collections import defaultdict

from shared import (
    load_session_data,
    group_by_session,
    ask_ollama,
    print_header,
    print_turn,
    Metrics,
)


SYSTEM_PROMPT = """You are an empathetic interview assistant conducting a qualitative 
interview. Use the structured entity context provided to ask personalised, 
connected follow-up questions. Reference specific people, emotions, and events 
the participant has mentioned."""

# Load spaCy model for entity extraction
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )


# ─────────────────────────────────────────────────────────
# MEMORY CLASS
# ─────────────────────────────────────────────────────────
class GraphMemory:
    """
    Builds a knowledge graph of entities and relationships from conversation turns.

    Node types : PERSON, EMOTION, TOPIC, EVENT, PLACE
    Edge types : MENTIONS, EVOKES, RELATES_TO, OCCURRED_IN
    """

    def __init__(self, session_name: str):
        self.session_name = session_name
        self.graph        = nx.DiGraph()
        self.turn_count   = 0

        # Simple keyword lists for emotion and topic detection
        self.emotion_keywords = {
            "guilt", "grief", "overwhelmed", "anxious", "sad", "angry",
            "relief", "love", "fear", "helpless", "proud", "lonely",
            "stressed", "grateful", "confused", "hopeful", "frustrated",
        }
        self.topic_keywords = {
            "eldercare", "caregiving", "academia", "tenure", "family",
            "hospital", "nursing home", "memory", "dementia", "work",
            "balance", "identity", "role", "responsibility", "support",
        }

    def _extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract named entities + keywords from text."""
        doc      = nlp(text.lower())
        entities = defaultdict(list)

        # spaCy named entities
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["PERSON"].append(ent.text.title())
            elif ent.label_ in ("GPE", "LOC", "FAC"):
                entities["PLACE"].append(ent.text.title())
            elif ent.label_ in ("EVENT", "WORK_OF_ART"):
                entities["EVENT"].append(ent.text.lower())

        # Keyword-based emotion + topic detection
        words = set(text.lower().split())
        for kw in self.emotion_keywords:
            if kw in words or kw in text.lower():
                entities["EMOTION"].append(kw)
        for kw in self.topic_keywords:
            if kw in words or kw in text.lower():
                entities["TOPIC"].append(kw)

        return dict(entities)

    def _add_to_graph(
        self,
        entities: dict[str, list[str]],
        turn_num: int,
        speaker: str,
    ):
        """Add extracted entities as nodes and create edges between co-occurring ones."""
        all_nodes = []

        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                node_id = f"{entity_type}::{entity}"
                if not self.graph.has_node(node_id):
                    self.graph.add_node(
                        node_id,
                        type=entity_type,
                        label=entity,
                        first_turn=turn_num,
                        speaker=speaker,
                        mentions=1,
                    )
                else:
                    self.graph.nodes[node_id]["mentions"] += 1
                all_nodes.append(node_id)

        # Connect all co-occurring entities in this turn
        for i, node_a in enumerate(all_nodes):
            for node_b in all_nodes[i + 1:]:
                if self.graph.has_edge(node_a, node_b):
                    self.graph[node_a][node_b]["weight"] += 1
                else:
                    self.graph.add_edge(
                        node_a, node_b,
                        relation="CO_OCCURS",
                        weight=1,
                        turn=turn_num,
                    )

    def store_turn(self, user_query: str, system_response: str):
        """Extract entities from both sides of a turn and add to graph."""
        self.turn_count += 1

        user_entities   = self._extract_entities(user_query)
        system_entities = self._extract_entities(system_response)

        self._add_to_graph(user_entities,   self.turn_count, "user")
        self._add_to_graph(system_entities, self.turn_count, "system")

    def retrieve_context(self, user_query: str) -> str:
        """
        Find entities in the query, then retrieve their graph neighbours.
        Returns a structured context string.
        """
        if len(self.graph.nodes) == 0:
            return ""

        query_entities = self._extract_entities(user_query)
        mentioned      = []
        for entity_list in query_entities.values():
            mentioned.extend(entity_list)

        if not mentioned:
            # Fallback: return the most frequently mentioned nodes
            top_nodes = sorted(
                self.graph.nodes(data=True),
                key=lambda x: x[1].get("mentions", 0),
                reverse=True,
            )[:5]
            context_lines = [
                f"- {d['type']}: '{d['label']}' (mentioned {d['mentions']}x)"
                for _, d in top_nodes
            ]
            return "Frequently discussed topics:\n" + "\n".join(context_lines)

        # For each matched entity, get its connected neighbours
        context_lines = []
        for entity in mentioned:
            for entity_type in ["PERSON", "EMOTION", "TOPIC", "EVENT", "PLACE"]:
                node_id = f"{entity_type}::{entity.lower()}"
                if self.graph.has_node(node_id):
                    neighbours = list(self.graph.neighbors(node_id))[:4]
                    neighbour_labels = [
                        self.graph.nodes[n]["label"] for n in neighbours
                    ]
                    node_data = self.graph.nodes[node_id]
                    context_lines.append(
                        f"- {node_data['type']} '{entity}': "
                        f"connected to {neighbour_labels} "
                        f"(first mentioned turn {node_data['first_turn']}, "
                        f"{node_data['mentions']}x total)"
                    )

        if not context_lines:
            return ""

        return "Known entities related to this topic:\n" + "\n".join(context_lines)

    def graph_stats(self) -> str:
        return (
            f"Graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

    def get_response(self, user_query: str) -> tuple[str, str]:
        """Retrieve graph context, build prompt, get LLM response."""
        context = self.retrieve_context(user_query)

        if context:
            augmented_query = (
                f"Structured context from the conversation graph:\n{context}\n\n"
                f"Current user message: {user_query}"
            )
        else:
            augmented_query = user_query

        messages = [{"role": "user", "content": augmented_query}]
        response = ask_ollama(messages, system_prompt=SYSTEM_PROMPT)

        self.store_turn(user_query, response)
        return response, context


# ─────────────────────────────────────────────────────────
# VALIDATE WITH SAMPLE DATA
# ─────────────────────────────────────────────────────────
def run_experiment(csv_path: str = "sample_data.csv"):
    print_header("M3 — Graph Store (NetworkX)")

    rows     = load_session_data(csv_path)
    sessions = group_by_session(rows)
    metrics  = Metrics("M3 - Graph Store")

    for session_name, turns in sessions.items():
        print(f"\n📂 Session: {session_name}  ({len(turns)} turns)")
        memory = GraphMemory(session_name)

        # Pre-load historical turns from CSV into the graph
        for turn in turns[:-1]:
            memory.store_turn(turn["user_query"], turn["system_response"])
        print(f"   Pre-loaded {memory.turn_count} turns.  {memory.graph_stats()}")

        # Test on each turn
        for i, turn in enumerate(turns, 1):
            user_query = turn["user_query"]

            start             = time.time()
            response, context = memory.get_response(user_query)
            latency           = time.time() - start

            metrics.record(response, latency)

            ground_truth = turn["system_response"]
            print_turn(
                turn_num     = i,
                user         = user_query,
                response     = response,
                context_info = f"{memory.graph_stats()} | latency={latency:.2f}s",
            )
            print(f"  GROUND TRUTH : {ground_truth[:200]}...")
            if context:
                print(f"  GRAPH CTX    : {context[:300]}...")

    metrics.print_summary()
    return metrics.summary()


if __name__ == "__main__":
    run_experiment()
