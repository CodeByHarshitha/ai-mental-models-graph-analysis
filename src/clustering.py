from pathlib import Path
from typing import Dict, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score


def cluster_mental_models(
    similarity_df: pd.DataFrame,
    participant_graphs: Dict[str, nx.Graph],
    processed_df: pd.DataFrame,
    reports_dir: Path,
    min_k: int = 2,
    max_k: int = 6,
) -> Tuple[pd.DataFrame, float]:
    """
    Clusters participants using Agglomerative Clustering on the precomputed
    mental-model distance matrix (1 - similarity).
    Automatically selects the optimal number of clusters k via Silhouette Score.
    If silhouette < 0.15, avoids forcing artificial clusters and assigns a single group.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    pids = list(similarity_df.index)
    sim_matrix = similarity_df.values
    dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 1.0)
    np.fill_diagonal(dist_matrix, 0.0)

    best_k = 1
    best_score = -1.0
    best_labels = np.zeros(len(pids), dtype=int)

    upper_k = min(max_k, len(pids) - 1)
    for k in range(min_k, upper_k + 1):
        model = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average",
        )
        labels = model.fit_predict(dist_matrix)
        score = float(silhouette_score(dist_matrix, labels, metric="precomputed"))
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    # Guardrail: do not force clusters if structural separation is weak
    if best_score < 0.15:
        best_k = 1
        best_labels = np.zeros(len(pids), dtype=int)

    # 2D Multidimensional Scaling (MDS) coordinates for visualization
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords_2d = mds.fit_transform(dist_matrix)

    # Identify top defining concepts per cluster to create human-readable labels
    cluster_top_concepts: Dict[int, str] = {}
    for cid in range(best_k):
        member_pids = [pids[i] for i, lbl in enumerate(best_labels) if lbl == cid]
        concept_counts: Dict[str, int] = {}
        for pid in member_pids:
            for node in participant_graphs[pid].nodes():
                if node != "AI":  # Exclude universal anchor to highlight distinguishing concepts
                    concept_counts[node] = concept_counts.get(node, 0) + 1
        top_3 = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        theme = " + ".join([c for c, _ in top_3]) if top_3 else "General AI"
        cluster_top_concepts[cid] = f"Cluster {cid + 1}: {theme}"

    meta_lookup = processed_df.set_index("participant_id")[["role_background", "concepts"]].to_dict("index")

    rows = []
    for idx, pid in enumerate(pids):
        cid = int(best_labels[idx])
        rows.append({
            "participant_id": pid,
            "role_background": meta_lookup[pid]["role_background"],
            "cluster_id": cid + 1,
            "cluster_label": cluster_top_concepts[cid],
            "concepts": meta_lookup[pid]["concepts"],
            "mds_x": round(float(coords_2d[idx, 0]), 4),
            "mds_y": round(float(coords_2d[idx, 1]), 4),
        })

    clusters_df = pd.DataFrame(rows)
    clusters_df.to_csv(reports_dir / "participant_clusters.csv", index=False)

    return clusters_df, round(best_score, 4)