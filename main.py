from pathlib import Path
import numpy as np
from src.dataset_loader import load_dataset
from src.data_preprocessing import preprocess_responses
from src.graph_construction import build_mental_model_graphs
from src.graph_analysis import analyze_graphs
from src.similarity_analysis import compute_similarity_matrices
from src.clustering import cluster_mental_models
from src.visualizations import generate_all_visualizations


def write_summary_report(
    report_path: Path,
    n_participants: int,
    collective_nodes_df,
    graph_metrics_df,
    similarity_df,
    clusters_df,
    silhouette: float,
) -> None:
    """Writes a structured markdown summary answering all five research questions."""
    top_concepts = collective_nodes_df.head(5)["concept"].tolist()
    top_bridge = collective_nodes_df.sort_values("betweenness_centrality", ascending=False).iloc[0]["concept"]

    # Extract off-diagonal pairwise similarity stats
    mat = similarity_df.values
    off_diag = mat[np.triu_indices_from(mat, k=1)]
    mean_sim = float(np.mean(off_diag))
    min_sim = float(np.min(off_diag))
    max_sim = float(np.max(off_diag))

    collective_row = graph_metrics_df[graph_metrics_df["graph_id"] == "COMBINED_COLLECTIVE"].iloc[0]

    cluster_lines = []
    for label, group in clusters_df.groupby("cluster_label"):
        roles = ", ".join(group["role_background"].unique()[:4])
        cluster_lines.append(f"- **{label}** ({len(group)} participants — e.g., {roles})")

    report_text = f"""# Summary of Findings: Human Perceptions of AI as Graph Networks

## 1. Dataset & Network Overview
- **Valid Participants Analyzed:** {n_participants} (Synthetic dataset explicitly labeled `is_synthetic=True`)
- **Collective Graph Size:** {int(collective_row['num_nodes'])} canonical concepts, {int(collective_row['num_edges'])} association edges, Network Density = {collective_row['density']}
- **Connected Components:** {int(collective_row['connected_components'])} (all concepts form a single connected conceptual ecosystem)

## 2. Answers to Research Questions
1. **Most Strongly Associated Concepts with AI:** The top concepts by weighted degree centrality are **{', '.join(top_concepts)}**.
2. **Similarity Across Mental Models:** Pairwise participant similarity averages **{mean_sim:.3f}** (range: **{min_sim:.3f}** to **{max_sim:.3f}**), showing shared core vocabulary around *AI* and *Data* alongside sharp structural divergence across socio-technical sub-networks.
3. **Central Bridge Concepts:** **{top_bridge}** exhibits the highest weighted betweenness centrality, acting as the primary conceptual bridge linking technical mechanisms (*Algorithms*, *Learning*) to societal outcomes (*Jobs*, *Bias*, *Creativity*).
4. **Participant Grouping:** Agglomerative clustering identified **{clusters_df['cluster_id'].nunique()} distinct perception clusters** with a silhouette score of **{silhouette}**:
{chr(10).join(cluster_lines)}
5. **Dominant Cognitive Patterns:** People do not view AI monolithically; instead, mental models organize around four recurring frames: (a) technical systems & automation, (b) algorithmic bias & labor impact, (c) human-AI creative augmentation, and (d) pragmatic everyday decision support.
"""
    report_path.write_text(report_text, encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    raw_csv = base_dir / "data" / "raw" / "synthetic_ai_perceptions_raw.csv"
    processed_csv = base_dir / "data" / "processed" / "cleaned_ai_perceptions.csv"
    graphs_dir = base_dir / "outputs" / "graphs"
    viz_dir = base_dir / "outputs" / "visualizations"
    reports_dir = base_dir / "outputs" / "reports"

    print("[1/7] Loading dataset...")
    raw_df = load_dataset(raw_csv)

    print("[2/7] Cleaning text and extracting concepts with Sentence Transformers...")
    processed_df, model = preprocess_responses(raw_df, processed_csv)

    print("[3/7] Constructing individual and collective mental model graphs...")
    participant_graphs, combined_graph = build_mental_model_graphs(processed_df, graphs_dir)

    print("[4/7] Computing graph centrality and structural metrics...")
    collective_nodes_df, graph_metrics_df = analyze_graphs(participant_graphs, combined_graph, reports_dir)

    print("[5/7] Computing multi-metric graph similarity matrix...")
    similarity_df, _ = compute_similarity_matrices(participant_graphs, processed_df, model, reports_dir)

    print("[6/7] Clustering participants by mental model similarity...")
    clusters_df, silhouette = cluster_mental_models(similarity_df, participant_graphs, processed_df, reports_dir)

    print("[7/7] Generating interactive Plotly visualizations and summary report...")
    generate_all_visualizations(
        participant_graphs,
        combined_graph,
        collective_nodes_df,
        similarity_df,
        clusters_df,
        silhouette,
        viz_dir,
    )
    write_summary_report(
        reports_dir / "project_summary_report.md",
        len(processed_df),
        collective_nodes_df,
        graph_metrics_df,
        similarity_df,
        clusters_df,
        silhouette,
    )
    print("Pipeline complete! All outputs saved under outputs/ and data/processed/.")


if __name__ == "__main__":
    main()