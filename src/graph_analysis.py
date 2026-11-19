from pathlib import Path
from typing import Dict, Tuple
import networkx as nx
import pandas as pd


def analyze_graphs(
    participant_graphs: Dict[str, nx.Graph],
    combined_graph: nx.Graph,
    reports_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes structural and centrality metrics for both the collective graph
    and individual participant mental model graphs.
    Avoids unreliable centrality calculations on tiny (< 3 node) individual graphs.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collective Graph Node Centrality Analysis
    deg_cent = nx.degree_centrality(combined_graph)
    # Invert weight so stronger association = shorter path distance for betweenness/closeness
    distance_graph = combined_graph.copy()
    for u, v, data in distance_graph.edges(data=True):
        data["distance"] = 1.0 / max(data["weight"], 1e-6)

    bet_cent = nx.betweenness_centrality(distance_graph, weight="distance", normalized=True)
    clo_cent = nx.closeness_centrality(distance_graph, distance="distance")

    weighted_degree = dict(combined_graph.degree(weight="weight"))

    node_rows = []
    for node, attrs in combined_graph.nodes(data=True):
        node_rows.append({
            "concept": node,
            "category": attrs.get("category", "Other"),
            "participant_frequency": attrs.get("participant_count", 0),
            "weighted_degree": round(weighted_degree[node], 3),
            "degree_centrality": round(deg_cent[node], 4),
            "betweenness_centrality": round(bet_cent[node], 4),
            "closeness_centrality": round(clo_cent[node], 4),
        })

    collective_nodes_df = (
        pd.DataFrame(node_rows)
        .sort_values(by=["weighted_degree", "betweenness_centrality"], ascending=False)
        .reset_index(drop=True)
    )
    collective_nodes_df.to_csv(reports_dir / "collective_node_metrics.csv", index=False)

    # 2. Individual & Collective Graph-Level Structural Metrics
    graph_rows = []
    all_graphs = {"COMBINED_COLLECTIVE": combined_graph, **participant_graphs}

    for gid, g in all_graphs.items():
        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()
        density = nx.density(g) if n_nodes > 1 else 0.0
        n_components = nx.number_connected_components(g) if n_nodes > 0 else 0

        # Betweenness is only mathematically meaningful when a graph has >= 3 nodes
        if n_nodes >= 3 and nx.is_connected(g):
            local_bet = nx.betweenness_centrality(g)
            top_bridge = max(local_bet, key=local_bet.get)
        else:
            top_bridge = "N/A (graph < 3 nodes or disconnected)"

        graph_rows.append({
            "graph_id": gid,
            "role_background": g.graph.get("role_background", "All Participants"),
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "density": round(density, 4),
            "connected_components": n_components,
            "top_bridge_concept": top_bridge,
        })

    graph_metrics_df = pd.DataFrame(graph_rows)
    graph_metrics_df.to_csv(reports_dir / "graph_level_metrics.csv", index=False)

    return collective_nodes_df, graph_metrics_df