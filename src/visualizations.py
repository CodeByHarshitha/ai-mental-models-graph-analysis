from pathlib import Path
from typing import Dict
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


CATEGORY_COLORS: Dict[str, str] = {
    "Core Entity": "#00D1FF",
    "Technical Infrastructure": "#3B82F6",
    "Application & Impact": "#10B981",
    "Socio-Economic Impact": "#F59E0B",
    "Ethics & Governance": "#EF4444",
    "Human Agency": "#A855F7",
    "Other": "#94A3B8",
}


def plot_combined_network(
    combined_graph: nx.Graph,
    collective_nodes_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Creates an interactive Plotly network graph of the collective AI mental model."""
    pos = nx.spring_layout(combined_graph, weight="weight", seed=42, k=0.9)
    metrics_lookup = collective_nodes_df.set_index("concept").to_dict("index")

    fig = go.Figure()

    # Draw weighted edges individually so line thickness scales with association strength
    max_weight = max((d["weight"] for _, _, d in combined_graph.edges(data=True)), default=1.0)
    for u, v, data in combined_graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data["weight"]
        line_width = 1.0 + (weight / max_weight) * 7.0
        opacity = 0.25 + (weight / max_weight) * 0.55

        fig.add_trace(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=line_width, color=f"rgba(148, 163, 184, {opacity:.2f})"),
                hoverinfo="text",
                text=f"{u} ↔ {v}<br>Association Weight: {weight:.1f} ({data['co_occurrence_count']} participants)",
                showlegend=False,
            )
        )

    # Group nodes by category for clean legend toggling
    categories = sorted({attrs.get("category", "Other") for _, attrs in combined_graph.nodes(data=True)})
    for cat in categories:
        cat_nodes = [n for n, attrs in combined_graph.nodes(data=True) if attrs.get("category", "Other") == cat]
        node_x = [pos[n][0] for n in cat_nodes]
        node_y = [pos[n][1] for n in cat_nodes]
        node_sizes = [18 + metrics_lookup[n]["weighted_degree"] * 1.4 for n in cat_nodes]
        hover_texts = [
            (
                f"<b>{n}</b> ({cat})<br>"
                f"Participants Mentioning: {metrics_lookup[n]['participant_frequency']}<br>"
                f"Weighted Degree: {metrics_lookup[n]['weighted_degree']}<br>"
                f"Betweenness Centrality: {metrics_lookup[n]['betweenness_centrality']}<br>"
                f"Closeness Centrality: {metrics_lookup[n]['closeness_centrality']}"
            )
            for n in cat_nodes
        ]

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                name=cat,
                text=cat_nodes,
                textposition="top center",
                textfont=dict(size=12, color="#F8FAFC"),
                hoverinfo="text",
                hovertext=hover_texts,
                marker=dict(
                    size=node_sizes,
                    color=CATEGORY_COLORS.get(cat, "#94A3B8"),
                    line=dict(width=2, color="#0F172A"),
                ),
            )
        )

    fig.update_layout(
        title="Collective Human Mental Model of Artificial Intelligence (Node Size = Weighted Centrality)",
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(title="Concept Domain", orientation="h", yanchor="bottom", y=-0.08, x=0.05),
        margin=dict(l=20, r=20, t=60, b=50),
        height=700,
    )
    fig.write_html(output_path)


def plot_sample_participant_graphs(
    participant_graphs: Dict[str, nx.Graph],
    clusters_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plots one representative individual mental model from each discovered perception cluster."""
    representatives = clusters_df.groupby("cluster_id").first().reset_index()
    n_panels = len(representatives)

    fig = make_subplots(
        rows=1,
        cols=n_panels,
        subplot_titles=[
            f"{row['participant_id']} ({row['role_background']})<br><sup>{row['cluster_label']}</sup>"
            for _, row in representatives.iterrows()
        ],
    )

    for col_idx, (_, row) in enumerate(representatives.iterrows(), start=1):
        pid = row["participant_id"]
        g = participant_graphs[pid]
        pos = nx.circular_layout(g)

        for u, v, data in g.edges(data=True):
            fig.add_trace(
                go.Scatter(
                    x=[pos[u][0], pos[v][0], None],
                    y=[pos[u][1], pos[v][1], None],
                    mode="lines",
                    line=dict(width=1.5 + data["weight"] * 2.0, color="#64748B"),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=col_idx,
            )

        nodes = list(g.nodes())
        fig.add_trace(
            go.Scatter(
                x=[pos[n][0] for n in nodes],
                y=[pos[n][1] for n in nodes],
                mode="markers+text",
                text=nodes,
                textposition="top center",
                marker=dict(
                    size=18,
                    color=[CATEGORY_COLORS.get(g.nodes[n].get("category", "Other"), "#94A3B8") for n in nodes],
                    line=dict(width=1.5, color="#FFFFFF"),
                ),
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )

    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-1.45, 1.45])
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, range=[-1.45, 1.45])
    fig.update_layout(
        title="Comparison of Individual Participant Mental Model Graphs Across Clusters",
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120",
        height=500,
        margin=dict(l=20, r=20, t=90, b=30),
    )
    fig.write_html(output_path)


def plot_similarity_heatmap(similarity_df: pd.DataFrame, output_path: Path) -> None:
    """Creates an interactive participant-by-participant similarity heatmap."""
    fig = px.imshow(
        similarity_df.values,
        x=similarity_df.columns,
        y=similarity_df.index,
        color_continuous_scale="Tealgrn",
        zmin=0.0,
        zmax=1.0,
        labels=dict(x="Participant ID", y="Participant ID", color="Combined Similarity"),
        title="Participant-by-Participant Mental Model Similarity Matrix",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120",
        height=720,
        width=820,
    )
    fig.write_html(output_path)


def plot_participant_clusters(
    clusters_df: pd.DataFrame,
    silhouette: float,
    output_path: Path,
) -> None:
    """Visualizes participant perception clusters in 2D MDS space."""
    fig = px.scatter(
        clusters_df,
        x="mds_x",
        y="mds_y",
        color="cluster_label",
        text="participant_id",
        hover_data=["role_background", "concepts"],
        title=f"Participant Mental Model Clusters (MDS Projection | Silhouette Score = {silhouette})",
        labels={"mds_x": "Cognitive Dimension 1", "mds_y": "Cognitive Dimension 2", "cluster_label": "Perception Group"},
    )
    fig.update_traces(textposition="top center", marker=dict(size=14, line=dict(width=1.5, color="#FFFFFF")))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120",
        height=620,
    )
    fig.write_html(output_path)


def generate_all_visualizations(
    participant_graphs: Dict[str, nx.Graph],
    combined_graph: nx.Graph,
    collective_nodes_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    silhouette: float,
    viz_dir: Path,
) -> None:
    """Generates all interactive HTML visualizations."""
    viz_dir.mkdir(parents=True, exist_ok=True)
    plot_combined_network(combined_graph, collective_nodes_df, viz_dir / "combined_ai_network.html")
    plot_sample_participant_graphs(participant_graphs, clusters_df, viz_dir / "sample_participant_graphs.html")
    plot_similarity_heatmap(similarity_df, viz_dir / "similarity_heatmap.html")
    plot_participant_clusters(clusters_df, silhouette, viz_dir / "participant_clusters.html")