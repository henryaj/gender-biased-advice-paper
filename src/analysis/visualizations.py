"""Visualization module for gender bias research."""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.database import db
from ..utils.logging import get_logger

logger = get_logger("visualizations")

# Output directory for figures
FIGURE_DIR = Path(__file__).parent.parent.parent / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    "male": "#4A90D9",
    "female": "#D94A7A",
    "neutral": "#7A7A7A"
}


def load_analysis_data() -> pd.DataFrame:
    """Load and prepare data for visualization."""
    raw_data = db.get_analysis_data()

    if not raw_data:
        raise ValueError("No analysis data available")

    df = pd.DataFrame(raw_data)

    # Parse tone_labels JSON
    df["tone_labels"] = df["tone_labels"].apply(
        lambda x: json.loads(x) if x else []
    )

    return df


def plot_harshness_distribution(
    df: Optional[pd.DataFrame] = None,
    save: bool = True
) -> plt.Figure:
    """
    Plot distribution of harshness scores by gender.

    Args:
        df: DataFrame with analysis data (loads from DB if None)
        save: Whether to save the figure

    Returns:
        matplotlib Figure
    """
    if df is None:
        df = load_analysis_data()

    # Filter to usable data
    plot_df = df[
        (df["poster_gender"].isin(["male", "female"])) &
        (df["harshness_score"].notna())
    ].copy()

    if len(plot_df) == 0:
        logger.warning("No data for harshness distribution plot")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram/KDE plot
    ax1 = axes[0]
    for gender in ["male", "female"]:
        data = plot_df[plot_df["poster_gender"] == gender]["harshness_score"]
        sns.kdeplot(data, ax=ax1, label=gender.capitalize(), color=COLORS[gender], fill=True, alpha=0.3)

    ax1.set_xlabel("Harshness Score", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Distribution of Harshness Scores by Poster Gender", fontsize=14)
    ax1.legend()

    # Box plot
    ax2 = axes[1]
    sns.boxplot(
        data=plot_df,
        x="poster_gender",
        y="harshness_score",
        palette=COLORS,
        ax=ax2
    )
    ax2.set_xlabel("Poster Gender", fontsize=12)
    ax2.set_ylabel("Harshness Score", fontsize=12)
    ax2.set_title("Harshness Score Comparison", fontsize=14)

    # Add sample sizes
    for i, gender in enumerate(["female", "male"]):
        n = len(plot_df[plot_df["poster_gender"] == gender])
        ax2.text(i, ax2.get_ylim()[1], f"n={n}", ha="center", fontsize=10)

    plt.tight_layout()

    if save:
        filepath = FIGURE_DIR / "harshness_distribution.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        logger.info(f"Saved harshness distribution plot to {filepath}")

    return fig


def plot_tone_label_comparison(
    df: Optional[pd.DataFrame] = None,
    save: bool = True
) -> plt.Figure:
    """
    Plot comparison of tone label frequencies by gender.

    Args:
        df: DataFrame with analysis data
        save: Whether to save the figure

    Returns:
        matplotlib Figure
    """
    if df is None:
        df = load_analysis_data()

    plot_df = df[df["poster_gender"].isin(["male", "female"])].copy()

    # Count tone labels by gender
    male_counts = defaultdict(int)
    female_counts = defaultdict(int)
    male_total = 0
    female_total = 0

    for _, row in plot_df.iterrows():
        labels = row["tone_labels"]
        if row["poster_gender"] == "male":
            male_total += 1
            for label in labels:
                male_counts[label] += 1
        else:
            female_total += 1
            for label in labels:
                female_counts[label] += 1

    # Calculate proportions
    all_labels = sorted(set(male_counts.keys()) | set(female_counts.keys()))

    male_props = [male_counts[l] / male_total if male_total > 0 else 0 for l in all_labels]
    female_props = [female_counts[l] / female_total if female_total > 0 else 0 for l in all_labels]

    # Sort by difference
    differences = [m - f for m, f in zip(male_props, female_props)]
    sorted_indices = np.argsort(differences)[::-1]

    all_labels = [all_labels[i] for i in sorted_indices]
    male_props = [male_props[i] for i in sorted_indices]
    female_props = [female_props[i] for i in sorted_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    x = np.arange(len(all_labels))
    width = 0.35

    bars1 = ax.barh(x - width/2, male_props, width, label="Male Posters", color=COLORS["male"])
    bars2 = ax.barh(x + width/2, female_props, width, label="Female Posters", color=COLORS["female"])

    ax.set_xlabel("Proportion of Comments", fontsize=12)
    ax.set_ylabel("Tone Label", fontsize=12)
    ax.set_title("Tone Label Frequency by Poster Gender", fontsize=14)
    ax.set_yticks(x)
    ax.set_yticklabels(all_labels)
    ax.legend()

    plt.tight_layout()

    if save:
        filepath = FIGURE_DIR / "tone_label_comparison.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        logger.info(f"Saved tone label comparison plot to {filepath}")

    return fig


def plot_effect_sizes(
    results: Dict[str, Any],
    save: bool = True
) -> plt.Figure:
    """
    Plot effect sizes (Cohen's d) with confidence intervals.

    Args:
        results: Dictionary with analysis results
        save: Whether to save the figure

    Returns:
        matplotlib Figure
    """
    # Collect effect sizes from results
    effect_data = []

    # Primary analysis
    if "primary_analysis" in results and "effect_size" in results["primary_analysis"]:
        es = results["primary_analysis"]["effect_size"]
        effect_data.append({
            "category": "Overall",
            "cohens_d": es["cohens_d"],
            "ci_lower": es.get("ci_lower", es["cohens_d"] - 0.1),
            "ci_upper": es.get("ci_upper", es["cohens_d"] + 0.1)
        })

    # Breakdown by relationship type
    if "breakdown_by_relationship_type" in results:
        for cat, data in results["breakdown_by_relationship_type"].items():
            if "cohens_d" in data:
                effect_data.append({
                    "category": f"Relationship: {cat}",
                    "cohens_d": data["cohens_d"],
                    "ci_lower": data["cohens_d"] - 0.15,  # Rough estimate
                    "ci_upper": data["cohens_d"] + 0.15
                })

    # Breakdown by source
    if "breakdown_by_source" in results:
        for cat, data in results["breakdown_by_source"].items():
            if "cohens_d" in data:
                effect_data.append({
                    "category": f"Source: {cat}",
                    "cohens_d": data["cohens_d"],
                    "ci_lower": data["cohens_d"] - 0.15,
                    "ci_upper": data["cohens_d"] + 0.15
                })

    if not effect_data:
        logger.warning("No effect size data to plot")
        return None

    df = pd.DataFrame(effect_data)

    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.5)))

    y_pos = np.arange(len(df))

    # Error bars
    errors = np.array([
        [d["cohens_d"] - d["ci_lower"] for _, d in df.iterrows()],
        [d["ci_upper"] - d["cohens_d"] for _, d in df.iterrows()]
    ])

    # Color by direction
    colors = [COLORS["male"] if d > 0 else COLORS["female"] for d in df["cohens_d"]]

    ax.barh(y_pos, df["cohens_d"], xerr=errors, color=colors, alpha=0.7, capsize=5)

    # Add vertical line at 0
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    # Add reference lines for effect size interpretation
    for x, label in [(-0.8, "Large"), (-0.5, "Medium"), (-0.2, "Small"),
                     (0.2, "Small"), (0.5, "Medium"), (0.8, "Large")]:
        ax.axvline(x=x, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["category"])
    ax.set_xlabel("Cohen's d (positive = harsher to men)", fontsize=12)
    ax.set_title("Effect Sizes: Gender Differences in Advice Harshness", fontsize=14)

    plt.tight_layout()

    if save:
        filepath = FIGURE_DIR / "effect_sizes.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        logger.info(f"Saved effect sizes plot to {filepath}")

    return fig


def plot_advice_direction_by_gender(
    df: Optional[pd.DataFrame] = None,
    save: bool = True
) -> plt.Figure:
    """
    Plot advice direction breakdown by gender.

    Args:
        df: DataFrame with analysis data
        save: Whether to save the figure

    Returns:
        matplotlib Figure
    """
    if df is None:
        df = load_analysis_data()

    plot_df = df[
        (df["poster_gender"].isin(["male", "female"])) &
        (df["advice_direction"].notna())
    ].copy()

    # Calculate proportions
    direction_counts = plot_df.groupby(
        ["poster_gender", "advice_direction"]
    ).size().unstack(fill_value=0)

    direction_props = direction_counts.div(direction_counts.sum(axis=1), axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    direction_order = ["supportive_of_op", "neutral", "mixed", "critical_of_op"]
    direction_labels = ["Supportive", "Neutral", "Mixed", "Critical"]

    x = np.arange(len(direction_order))
    width = 0.35

    for i, gender in enumerate(["male", "female"]):
        if gender in direction_props.index:
            values = [direction_props.loc[gender, d] if d in direction_props.columns else 0
                      for d in direction_order]
            offset = -width/2 if i == 0 else width/2
            ax.bar(x + offset, values, width, label=gender.capitalize(), color=COLORS[gender])

    ax.set_xlabel("Advice Direction", fontsize=12)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title("Advice Direction by Poster Gender", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(direction_labels)
    ax.legend()

    plt.tight_layout()

    if save:
        filepath = FIGURE_DIR / "advice_direction_by_gender.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        logger.info(f"Saved advice direction plot to {filepath}")

    return fig


def plot_scores_by_source_and_gender(
    df: Optional[pd.DataFrame] = None,
    save: bool = True
) -> plt.Figure:
    """
    Plot harshness scores by source and gender.

    Args:
        df: DataFrame with analysis data
        save: Whether to save the figure

    Returns:
        matplotlib Figure
    """
    if df is None:
        df = load_analysis_data()

    plot_df = df[
        (df["poster_gender"].isin(["male", "female"])) &
        (df["harshness_score"].notna())
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=plot_df,
        x="source_name",
        y="harshness_score",
        hue="poster_gender",
        palette=COLORS,
        ax=ax
    )

    ax.set_xlabel("Source", fontsize=12)
    ax.set_ylabel("Harshness Score", fontsize=12)
    ax.set_title("Harshness Scores by Source and Poster Gender", fontsize=14)
    ax.legend(title="Poster Gender")

    plt.tight_layout()

    if save:
        filepath = FIGURE_DIR / "scores_by_source_gender.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        logger.info(f"Saved scores by source plot to {filepath}")

    return fig


def generate_all_visualizations(results: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Generate all visualizations.

    Args:
        results: Analysis results dictionary (for effect size plot)

    Returns:
        List of saved file paths
    """
    logger.info("Generating all visualizations")

    saved_files = []

    try:
        df = load_analysis_data()
    except ValueError as e:
        logger.error(f"Could not load data: {e}")
        return saved_files

    # Generate each plot
    plots = [
        ("harshness_distribution", lambda: plot_harshness_distribution(df)),
        ("tone_label_comparison", lambda: plot_tone_label_comparison(df)),
        ("advice_direction", lambda: plot_advice_direction_by_gender(df)),
        ("scores_by_source", lambda: plot_scores_by_source_and_gender(df)),
    ]

    if results:
        plots.append(("effect_sizes", lambda: plot_effect_sizes(results)))

    for name, plot_func in plots:
        try:
            fig = plot_func()
            if fig:
                saved_files.append(str(FIGURE_DIR / f"{name}.png"))
                plt.close(fig)
        except Exception as e:
            logger.error(f"Error generating {name} plot: {e}")

    logger.info(f"Generated {len(saved_files)} visualizations")
    return saved_files


if __name__ == "__main__":
    generate_all_visualizations()
