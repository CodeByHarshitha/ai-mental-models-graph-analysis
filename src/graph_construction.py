from pathlib import Path
from typing import Dict, Tuple
import networkx as nx
import pandas as pd


CONCEPT_CATEGORIES: Dict[str, str] = {
    "AI": "Core Entity",
    "Intelligence": "Core Entity",
    "Data": "Technical Infrastructure",
    "Algorithms": "Technical Infrastructure",
    "Learning": "Technical Infrastructure",
    "Automation": "Application & Impact",
    "Decision-Making": "Application & Impact",
    "Jobs": "Socio-Economic Impact",
    "Bias": "Ethics & Governance",
    "Trust": "Ethics & Governance",
    "Humans": "Human Agency",
    "Creativity": "Human Agency",
}


def build_mental_model_graphs(
    processed_df: pd.DataFrame,
    graphs_output_dir: Path,
) -> Tuple[Dict[str, nx.Graph], nx.Graph]:
    """
    Constructs:
    1. Individual participant mental model graphs (undirected weighted graphs).
    2. A combined collective perception network aggregating node frequencies and edge weights.
    Saves GraphML exports for interoperability.
    """
    participant_dir = graphs_output_dir / "participant_graphs"
    participant_dir.mkdir(parents=True, exist_ok=True)

    participant_graphs: Dict[str, nx.Graph] = {}
    combined_graph = nx.Graph()

    for _, row in processed_df.iterrows():
        pid = str(row["participant_id"])
        role = str(row["role_background"])
        concepts = row["concepts"].split("|")
        raw_edges = row["concept_edges"].split("|") if pd.notna(row["concept_edges"]) and row["concept_edges"] else []

        g = nx.Graph(participant_id=pid, role_background=role)

        for concept in concepts:
            category = CONCEPT_CATEGORIES.get(concept, "Other")
            g.add_node(concept, category=category)

            if not combined_graph.has_node(concept):
                combined_graph.add_node(concept, category=category, participant_count=0)
            combined_graph.nodes[concept]["participant_count"] += 1

        for edge_str in raw_edges:
            if not edge_str:
                continue
            pair_part, weight_str = edge_str.split(":")
            u, v = pair_part.split("--")
            weight = float(weight_str)

            g.add_edge(u, v, weight=weight)

            if combined_graph.has_edge(u, v):
                combined_graph[u][v]["weight"] += weight
                combined_graph[u][v]["co_occurrence_count"] += 1
            else:
                combined_graph.add_edge(u, v, weight=weight, co_occurrence_count=1)

        participant_graphs[pid] = g
        nx.write_graphml(g, participant_dir / f"{pid}_mental_model.graphml")

    nx.write_graphml(combined_graph, graphs_output_dir / "combined_perception_network.graphml")
    return participant_graphs, combined_graph