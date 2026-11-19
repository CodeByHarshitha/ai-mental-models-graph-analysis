import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


# Canonical mental-model concepts and lexical trigger phrases
CANONICAL_CONCEPTS: Dict[str, List[str]] = {
    "AI": ["ai", "artificial intelligence", "generative ai", "machine intelligence"],
    "Humans": ["human", "humans", "people", "worker", "workers", "doctors", "staff", "empathy", "judgment"],
    "Data": ["data", "datasets", "training data", "information", "spreadsheet", "patterns"],
    "Algorithms": ["algorithm", "algorithms", "algorithmic", "neural network", "neural networks", "models", "software"],
    "Automation": ["automation", "automate", "automates", "automated", "routine", "repetitive"],
    "Creativity": ["creativity", "creative", "art", "music", "design", "brainstorm", "ideation"],
    "Jobs": ["job", "jobs", "employment", "workplace", "hiring", "career"],
    "Bias": ["bias", "fairness", "inequality", "errors", "accountability"],
    "Decision-Making": ["decision-making", "decision", "decisions", "choices", "oversight", "control", "governance"],
    "Learning": ["learning", "learns", "trained", "machine learning", "recognition"],
    "Intelligence": ["intelligence", "smart", "mimics", "simulate"],
    "Trust": ["trust", "privacy", "transparency", "truth", "integrity", "regulation"],
}


def normalize_text(text: str) -> str:
    """Lowercases text, normalizes hyphens, and strips non-essential punctuation."""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_concepts_and_pairs(
    cleaned_text: str,
    model: SentenceTransformer,
    concept_embeddings: np.ndarray,
    concept_names: List[str],
    semantic_threshold: float = 0.52,
) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """
    Extracts ordered canonical concepts from a participant's response using:
    1. Exact lexical keyword matching (preserving order of appearance in thought).
    2. Sentence-Transformer semantic similarity fallback on noun/phrase chunks
       to capture implicit conceptual associations.
    Returns ordered unique concepts and sequential/co-occurrence concept pairs.
    """
    found_positions: Dict[str, int] = {}

    # 1. Lexical phrase matching with character offset tracking
    for concept, triggers in CANONICAL_CONCEPTS.items():
        for trigger in triggers:
            match = re.search(rf"\b{re.escape(trigger)}\b", cleaned_text)
            if match:
                pos = match.start()
                if concept not in found_positions or pos < found_positions[concept]:
                    found_positions[concept] = pos

    # 2. Semantic matching via Sentence Transformers for uncaptured phrases
    chunks = [c.strip() for c in re.split(r"\b(?:and|while|though|yet|to|with|for|that|which)\b", cleaned_text) if len(c.strip()) > 3]
    if chunks:
        chunk_embs = model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
        sim_matrix = util.cos_sim(chunk_embs, concept_embeddings).cpu().numpy()
        for chunk_idx, scores in enumerate(sim_matrix):
            best_idx = int(np.argmax(scores))
            if scores[best_idx] >= semantic_threshold:
                matched_concept = concept_names[best_idx]
                if matched_concept not in found_positions:
                    found_positions[matched_concept] = 1000 + chunk_idx

    # Always anchor "AI" if implied by prompt context
    if "AI" not in found_positions:
        found_positions["AI"] = 0

    ordered_concepts = [c for c, _ in sorted(found_positions.items(), key=lambda item: item[1])]

    # Build directed/undirected association pairs based on narrative proximity
    # Adjacent concepts in thought receive weight 1.0; 2-step proximity receives 0.5
    edges: List[Tuple[str, str, float]] = []
    n = len(ordered_concepts)
    for i in range(n):
        for j in range(i + 1, n):
            distance = j - i
            if distance == 1:
                weight = 1.0
            elif distance == 2:
                weight = 0.5
            else:
                continue
            u, v = sorted([ordered_concepts[i], ordered_concepts[j]])
            edges.append((u, v, weight))

    return ordered_concepts, edges


def preprocess_responses(
    raw_df: pd.DataFrame,
    processed_csv_path: Path,
    model_name: str = "all-MiniLM-L6-v2",
) -> Tuple[pd.DataFrame, SentenceTransformer]:
    """
    Cleans raw survey responses, handles missing/duplicate entries, extracts concepts,
    and saves the processed table to CSV.
    """
    processed_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = raw_df.copy()
    # Handle missing values and blank strings
    df = df.dropna(subset=["participant_id", "open_response"])
    df["open_response"] = df["open_response"].astype(str).str.strip()
    df = df[df["open_response"].str.len() > 0]

    # Remove duplicate participant IDs (keep first valid entry)
    df = df.drop_duplicates(subset=["participant_id"], keep="first").reset_index(drop=True)
    df["role_background"] = df["role_background"].fillna("Unspecified")
    df["cleaned_text"] = df["open_response"].apply(normalize_text)

    # Load SentenceTransformer once and pre-encode canonical concept descriptions
    model = SentenceTransformer(model_name)
    concept_names = list(CANONICAL_CONCEPTS.keys())
    concept_descriptions = [
        f"{name}: {', '.join(triggers[:4])}" for name, triggers in CANONICAL_CONCEPTS.items()
    ]
    concept_embeddings = model.encode(concept_descriptions, convert_to_tensor=True, show_progress_bar=False)

    extracted_concepts_col = []
    extracted_edges_col = []

    for text in df["cleaned_text"]:
        concepts, edges = extract_concepts_and_pairs(
            text, model, concept_embeddings, concept_names
        )
        extracted_concepts_col.append("|".join(concepts))
        edge_strs = [f"{u}--{v}:{w}" for u, v, w in edges]
        extracted_edges_col.append("|".join(edge_strs))

    df["concepts"] = extracted_concepts_col
    df["concept_edges"] = extracted_edges_col

    # Keep participants with at least 2 concepts so a valid graph edge exists
    df = df[df["concepts"].apply(lambda x: len(x.split("|")) >= 2)].reset_index(drop=True)
    df.to_csv(processed_csv_path, index=False)

    return df, model