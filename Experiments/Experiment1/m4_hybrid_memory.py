"""
m4_hybrid_memory.py
───────────────────
EXPERIMENT M4 — Hybrid Memory (ChromaDB Vector + NetworkX Graph)

Combines both M2 and M3:
  - ChromaDB for semantic retrieval of similar past conversation turns
  - NetworkX graph for structured entity/relationship context

Both sources are retrieved for each query and merged into one context block
before calling the LLM.

What this measures:
  - Does combining semantic + structured context outperform either alone?
  - Does the extra context help or hurt (token overload / noise)?
  - Best-of-both-worlds: raw recall (vector) + relational depth (graph)

Install:
    pip install chromadb sentence-transformers networkx spacy
    python -m spacy download en_core_web_sm

Run:
    python m4_hybrid_memory.py
"""

import time
import uuid
import spacy
import networkx as nx
import chromadb
from chromadb.utils import embedding_functions
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
interview. You have access to both semantically similar past exchanges AND a 
structured knowledge graph of entities discussed. Use both to generate deeply 
personalised, contextually rich follow-up questions."""

TOP_K = 3  # vector store: number of similar turns to retrieve


# ─────────────────────────────────────────────────────────
# VECTOR STORE COMPONENT
# ─────────────────────────────────────────────────────────
class _VectorStore:
    def __init__(self, session_name: str):
        self.client    = chromadb.Client()
        self.embed_fn  = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        safe = session_name.replace(" ", "_").lower()
        self.collection = self.client.create_collection(
            name=f"hybrid_{safe}_{uuid.uuid4().hex[:6]}",
            embedding_function=self.embed_fn,
        )
        self.count = 0

    def store(self, user_query: str, system_response: str):
        self.count += 1
        self.collection.add(
            documents=[f"User: {user_query}\nAssistant: {system_response}"],
            ids=[f"turn_{self.count}"],
            metadatas=[{"turn": self.count}],
        )

    def retrieve(self, query: str) -> str:
        if self.count == 0:
            return ""
        k       = min(TOP_K, self.count)
        results = self.collection.query(query_texts=[query], n_results=k)
        docs    = results["documents"][0]
        return "\n\n".join(f"[Similar turn {i+1}]: {d}" for i, d in enumerate(docs))


# ─────────────────────────────────────────────────────────
# GRAPH STORE COMPONENT
# ─────────────────────────────────────────────────────────
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("Run: python -m spacy download en_core_web_sm")


class _GraphStore:
    EMOTIONS = {
        "guilt", "grief", "overwhelmed", "anxious", "sad", "angry",
        "relief", "love", "fear", "helpless", "proud", "lonely",
        "stressed", "grateful", "confused", "hopeful", "frustrated",
    }
    TOPICS = {
        "eldercare", "caregiving", "academia", "tenure", "family",
        "hospital", "nursing home", "memory", "dementia", "work",
        "balance", "identity", "role", "responsibility", "support",
    }

    def __init__(self):
        self.graph     = nx.DiGraph()
        self.turn_count = 0

    def _extract(self, text: str) -> dict[str, list[str]]:
        doc = _nlp(text.lower())
        ents = defaultdict(list)
        for e in doc.ents:
            if e.label_ == "PERSON":
                ents["PERSON"].append(e.text.title())
            elif e.label_ in ("GPE", "LOC"):
                ents["PLACE"].append(e.text.title())
        words = set(text.lower().split())
        for kw in self.EMOTIONS:
            if kw in text.lower():
                ents["EMOTION"].append(kw)
        for kw in self.TOPICS:
            if kw in words or kw in text.lower():
                ents["TOPIC"].append(kw)
        return dict(ents)

    def store(self, user_query: str, system_response: str):
        self.turn_count += 1
        for text, speaker in [(user_query, "user"), (system_response, "system")]:
            ents     = self._extract(text)
            all_nodes = []
            for etype, elist in ents.items():
                for label in elist:
                    nid = f"{etype}::{label}"
                    if not self.graph.has_node(nid):
                        self.graph.add_node(nid, type=etype, label=label,
                                            first_turn=self.turn_count,
                                            speaker=speaker, mentions=1)
                    else:
                        self.graph.nodes[nid]["mentions"] += 1
                    all_nodes.append(nid)
            for i, a in enumerate(all_nodes):
                for b in all_nodes[i + 1:]:
                    if self.graph.has_edge(a, b):
                        self.graph[a][b]["weight"] += 1
                    else:
                        self.graph.add_edge(a, b, relation="CO_OCCURS",
                                            weight=1, turn=self.turn_count)

    def retrieve(self, user_query: str) -> str:
        if not self.graph.nodes:
            return ""
        ents      = self._extract(user_query)
        mentioned = [e for elist in ents.values() for e in elist]
        lines     = []
        for entity in mentioned:
            for etype in ["PERSON", "EMOTION", "TOPIC", "PLACE"]:
                nid = f"{etype}::{entity.lower()}"
                if self.graph.has_node(nid):
                    neighbours = [
                        self.graph.nodes[n]["label"]
                        for n in list(self.graph.neighbors(nid))[:3]
                    ]
                    d = self.graph.nodes[nid]
                    lines.append(
                        f"- {d['type']} '{entity}' → related: {neighbours} "
                        f"(turn {d['first_turn']}, {d['mentions']}x)"
                    )
        return ("Graph entities:\n" + "\n".join(lines)) if lines else ""

    def stats(self) -> str:
        return (f"{self.graph.number_of_nodes()} nodes, "
                f"{self.graph.number_of_edges()} edges")


# ─────────────────────────────────────────────────────────
# HYBRID MEMORY CLASS
# ─────────────────────────────────────────────────────────
class HybridMemory:
    """Combines semantic vector retrieval with structured graph context."""

    def __init__(self, session_name: str):
        self.vector = _VectorStore(session_name)
        self.graph  = _GraphStore()

    def store_turn(self, user_query: str, system_response: str):
        self.vector.store(user_query, system_response)
        self.graph.store(user_query, system_response)

    def get_response(self, user_query: str) -> tuple[str, str, str]:
        """
        Returns (response, vector_context, graph_context)
        """
        vector_ctx = self.vector.retrieve(user_query)
        graph_ctx  = self.graph.retrieve(user_query)

        # Merge both context sources
        context_parts = []
        if vector_ctx:
            context_parts.append(f"=== Semantically Similar Past Turns ===\n{vector_ctx}")
        if graph_ctx:
            context_parts.append(f"=== Structured Entity Graph ===\n{graph_ctx}")

        merged_context = "\n\n".join(context_parts)

        if merged_context:
            augmented_query = (
                f"{merged_context}\n\n"
                f"Current user message: {user_query}"
            )
        else:
            augmented_query = user_query

        messages = [{"role": "user", "content": augmented_query}]
        response = ask_ollama(messages, system_prompt=SYSTEM_PROMPT)

        self.store_turn(user_query, response)
        return response, vector_ctx, graph_ctx


# ─────────────────────────────────────────────────────────
# VALIDATE WITH SAMPLE DATA
# ─────────────────────────────────────────────────────────
def run_experiment(csv_path: str = "sample_data.csv"):
    print_header("M4 — Hybrid Memory (Vector + Graph)")

    rows     = load_session_data(csv_path)
    sessions = group_by_session(rows)
    metrics  = Metrics("M4 - Hybrid Memory")

    for session_name, turns in sessions.items():
        print(f"\n📂 Session: {session_name}  ({len(turns)} turns)")
        memory = HybridMemory(session_name)

        # Pre-load historical turns
        for turn in turns[:-1]:
            memory.store_turn(turn["user_query"], turn["system_response"])
        print(f"   Pre-loaded {memory.graph.turn_count} turns.  "
              f"Graph: {memory.graph.stats()}")

        for i, turn in enumerate(turns, 1):
            user_query = turn["user_query"]

            start                         = time.time()
            response, v_ctx, g_ctx        = memory.get_response(user_query)
            latency                       = time.time() - start

            metrics.record(response, latency)

            ground_truth  = turn["system_response"]
            v_hits        = v_ctx.count("[Similar turn") if v_ctx else 0
            g_hits        = g_ctx.count("- ") if g_ctx else 0

            print_turn(
                turn_num     = i,
                user         = user_query,
                response     = response,
                context_info = (
                    f"Vector: {v_hits} turns | Graph: {g_hits} entities | "
                    f"latency={latency:.2f}s"
                ),
            )
            print(f"  GROUND TRUTH : {ground_truth[:200]}...")
            if v_ctx:
                print(f"  VECTOR CTX   : {v_ctx[:200]}...")
            if g_ctx:
                print(f"  GRAPH CTX    : {g_ctx[:200]}...")

    metrics.print_summary()
    return metrics.summary()


if __name__ == "__main__":
    run_experiment()
