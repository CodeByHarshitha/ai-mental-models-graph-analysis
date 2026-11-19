from pathlib import Path
from typing import Dict, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


def compute_jaccard_node_similarity(g1: nx.Graph, g2: nx.Graph) -> float:
    """Jaccard similarity |V1 ∩ V2| / |V1 ∪ V2| between concept node sets."""
    nodes1, nodes2 = set(g1.nodes()), set(g2.nodes())
    union = nodes1 | nodes2
    if not union:
        return 0.0
    return len(nodes1 & nodes2) / len(union)


def compute_edge_similarity(g1: nx.Graph, g2: nx.Graph) -> float:
    """Jaccard similarity |E1 ∩ E2| / |E1 ∪ E2| between undirected concept associations."""
    edges1 = {tuple(sorted(e)) for e in g1.edges()}
    edges2 = {tuple(sorted(e)) for e in g2.edges()}
    union = edges1 | edges2
    if not union:
        return 0.0
    return len(edges1 & edges2) / len(union)


def compute_similarity_matrices(
    participant_graphs: Dict[str, nx.Graph],
    processed_df: pd.DataFrame,
    model: SentenceTransformer,
    reports_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Computes four participant-by-participant similarity matrices:
    1. Node Jaccard Similarity (shared vocabulary of AI concepts)
    2. Edge Jaccard Similarity (shared structural links between concepts)
    3. Semantic Embedding Similarity (cosine similarity of full conceptual framing)
    4. Combined Weighted Mental Model Similarity:
       0.35 * Node_Jaccard + 0.35 * Edge_Jaccard + 0.30 * Semantic_Cosine
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    pids = list(participant_graphs.keys())
    n = len(pids)

    # Encode each participant's extracted concept + relationship narrative
    id_to_narrative = {}
    for _, row in processed_df.iterrows():
        pid = str(row["participant_id"])
        concepts_str = ", ".join(row["concepts"].split("|"))
        id_to_narrative[pid] = f"AI mental model connecting: {concepts_str}. Context: {row['cleaned_text']}"

    narratives = [id_to_narrative[pid] for pid in pids]
    embeddings = model.encode(narratives, convert_to_tensor=True, show_progress_bar=False)
    semantic_mat = np.clip(util.cos_sim(embeddings, embeddings).cpu().numpy(), 0.0, 1.0)

    node_mat = np.zeros((n, n))
    edge_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                node_mat[i, j] = 1.0
                edge_mat[i, j] = 1.0
            elif i < j:
                g1, g2 = participant_graphs[pids[i]], participant_graphs[pids[j]]
                n_sim = compute_jaccard_node_similarity(g1, g2)
                e_sim = compute_edge_similarity(g1, g2)
                node_mat[i, j] = node_mat[j, i] = n_sim
                edge_mat[i, j] = edge_mat[j, i] = e_sim

    combined_mat = (0.35 * node_mat) + (0.35 * edge_mat) + (0.30 * semantic_mat)
    np.fill_diagonal(combined_mat, 1.0)

    combined_df = pd.DataFrame(combined_mat, index=pids, columns=pids).round(4)
    combined_df.to_csv(reports_dir / "similarity_matrix.csv")

    component_dfs = {
        "node_jaccard": pd.DataFrame(node_mat, index=pids, columns=pids).round(4),
        "edge_jaccard": pd.DataFrame(edge_mat, index=pids, columns=pids).round(4),
        "semantic_cosine": pd.DataFrame(semantic_mat, index=pids, columns=pids).round(4),
        "combined": combined_df,
    }

    return combined_df, component_dfs