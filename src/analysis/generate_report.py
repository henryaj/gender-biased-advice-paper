"""Report generator for gender bias research analysis.

Generates tables, charts, and HTML report from the research dataset.
"""

import json
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from ..utils.database import db
from ..utils.logging import get_logger

logger = get_logger("generate_report")

# Output directories
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Ensure directories exist
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    "male": "#4A90D9",
    "female": "#D94A7A",
    "unknown": "#7A7A7A",
    "non-binary": "#9B59B6"
}
SEVERITY_COLORS = {"low": "#27ae60", "medium": "#f39c12", "high": "#e74c3c"}
FAULT_COLORS = {"none": "#27ae60", "some": "#f39c12", "substantial": "#e74c3c", "unclear": "#95a5a6"}


def load_post_data() -> pd.DataFrame:
    """Load post classification data."""
    with db.get_connection() as conn:
        query = """
            SELECT p.post_id, p.title, p.author, p.timestamp,
                   pc.poster_gender, pc.gender_confidence, pc.relationship_type,
                   pc.situation_severity, pc.op_fault, pc.problem_category,
                   s.name as source_name
            FROM posts p
            JOIN post_classifications pc ON p.post_id = pc.post_id
            LEFT JOIN sources s ON p.source_id = s.source_id
            WHERE pc.is_relationship_advice = 1
        """
        df = pd.read_sql_query(query, conn)
    return df


def load_comment_data() -> pd.DataFrame:
    """Load comment classification data with post info."""
    with db.get_connection() as conn:
        query = """
            SELECT c.comment_id, c.post_id, c.author, c.score,
                   cc.advice_direction, cc.tone_labels,
                   pc.poster_gender, pc.situation_severity, pc.op_fault
            FROM comments c
            JOIN comment_classifications cc ON c.comment_id = cc.comment_id
            JOIN post_classifications pc ON c.post_id = pc.post_id
            WHERE pc.is_relationship_advice = 1
        """
        df = pd.read_sql_query(query, conn)

        # Parse tone_labels JSON
        df["tone_labels"] = df["tone_labels"].apply(
            lambda x: json.loads(x) if x else []
        )
    return df


# =============================================================================
# TABLE GENERATION FUNCTIONS
# =============================================================================

def generate_dataset_summary() -> pd.DataFrame:
    """Generate T1: Dataset summary table."""
    with db.get_connection() as conn:
        stats = {}
        stats["Total Posts"] = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        stats["Total Comments"] = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
        stats["Classified Posts"] = conn.execute(
            "SELECT COUNT(*) FROM post_classifications"
        ).fetchone()[0]
        stats["Relationship Posts"] = conn.execute(
            "SELECT COUNT(*) FROM post_classifications WHERE is_relationship_advice = 1"
        ).fetchone()[0]
        stats["Classified Comments"] = conn.execute(
            "SELECT COUNT(*) FROM comment_classifications"
        ).fetchone()[0]
        stats["Pairwise Comparisons"] = conn.execute(
            "SELECT COUNT(*) FROM pairwise_comparisons"
        ).fetchone()[0]

    df = pd.DataFrame([
        {"Metric": k, "Count": v} for k, v in stats.items()
    ])

    # Save to CSV
    df.to_csv(TABLE_DIR / "dataset_summary.csv", index=False)
    logger.info(f"Saved dataset_summary.csv")

    return df


def generate_gender_table() -> pd.DataFrame:
    """Generate T2: Gender distribution table."""
    post_df = load_post_data()

    gender_counts = post_df["poster_gender"].value_counts()
    total = len(post_df)

    rows = []
    for gender in ["male", "female", "unknown", "non-binary"]:
        count = gender_counts.get(gender, 0)
        pct = (count / total * 100) if total > 0 else 0
        avg_conf = post_df[post_df["poster_gender"] == gender]["gender_confidence"].mean()
        rows.append({
            "Gender": gender.capitalize(),
            "Count": count,
            "Percentage": f"{pct:.1f}%",
            "Avg Confidence": f"{avg_conf:.1%}" if pd.notna(avg_conf) else "N/A"
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "gender_distribution.csv", index=False)
    logger.info(f"Saved gender_distribution.csv")

    return df


def generate_confound_balance_table() -> pd.DataFrame:
    """Generate T3: Confound balance table with chi-square tests."""
    post_df = load_post_data()

    # Filter to known genders for chi-square
    known_df = post_df[post_df["poster_gender"].isin(["male", "female"])]

    results = []

    # Test severity by gender
    if len(known_df) > 0 and known_df["situation_severity"].notna().sum() > 0:
        contingency = pd.crosstab(known_df["poster_gender"], known_df["situation_severity"])
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            results.append({
                "Confound": "Severity",
                "Chi-square": f"{chi2:.2f}",
                "p-value": f"{p:.4f}",
                "Significant": "Yes" if p < 0.05 else "No",
                "Note": "Low n" if contingency.min().min() < 5 else ""
            })

    # Test fault by gender
    if len(known_df) > 0 and known_df["op_fault"].notna().sum() > 0:
        contingency = pd.crosstab(known_df["poster_gender"], known_df["op_fault"])
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            results.append({
                "Confound": "OP Fault",
                "Chi-square": f"{chi2:.2f}",
                "p-value": f"{p:.4f}",
                "Significant": "Yes" if p < 0.05 else "No",
                "Note": "Low n" if contingency.min().min() < 5 else ""
            })

    # Test problem category by gender
    if len(known_df) > 0 and known_df["problem_category"].notna().sum() > 0:
        contingency = pd.crosstab(known_df["poster_gender"], known_df["problem_category"])
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            results.append({
                "Confound": "Problem Category",
                "Chi-square": f"{chi2:.2f}",
                "p-value": f"{p:.4f}",
                "Significant": "Yes" if p < 0.05 else "No",
                "Note": "Low n" if contingency.min().min() < 5 else ""
            })

    df = pd.DataFrame(results) if results else pd.DataFrame(
        columns=["Confound", "Chi-square", "p-value", "Significant", "Note"]
    )
    df.to_csv(TABLE_DIR / "confound_balance.csv", index=False)
    logger.info(f"Saved confound_balance.csv")

    return df


def generate_tone_frequency_table() -> pd.DataFrame:
    """Generate T4: Tone label frequency table."""
    comment_df = load_comment_data()

    if len(comment_df) == 0:
        df = pd.DataFrame(columns=["Tone", "Male Count", "Female Count", "Male %", "Female %"])
        df.to_csv(TABLE_DIR / "tone_frequencies.csv", index=False)
        return df

    # Count tones by gender
    male_counts = defaultdict(int)
    female_counts = defaultdict(int)
    male_total = len(comment_df[comment_df["poster_gender"] == "male"])
    female_total = len(comment_df[comment_df["poster_gender"] == "female"])

    for _, row in comment_df.iterrows():
        for tone in row["tone_labels"]:
            if row["poster_gender"] == "male":
                male_counts[tone] += 1
            elif row["poster_gender"] == "female":
                female_counts[tone] += 1

    all_tones = sorted(set(male_counts.keys()) | set(female_counts.keys()))

    rows = []
    for tone in all_tones:
        mc = male_counts[tone]
        fc = female_counts[tone]
        rows.append({
            "Tone": tone,
            "Male Count": mc,
            "Female Count": fc,
            "Male %": f"{mc/male_total*100:.1f}%" if male_total > 0 else "0%",
            "Female %": f"{fc/female_total*100:.1f}%" if female_total > 0 else "0%"
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "tone_frequencies.csv", index=False)
    logger.info(f"Saved tone_frequencies.csv")

    return df


def generate_advice_direction_table() -> pd.DataFrame:
    """Generate T5: Advice direction by gender table."""
    comment_df = load_comment_data()

    if len(comment_df) == 0:
        df = pd.DataFrame(columns=["Direction", "Male Count", "Female Count", "Male %", "Female %"])
        df.to_csv(TABLE_DIR / "advice_direction.csv", index=False)
        return df

    # Cross-tabulation
    known_df = comment_df[comment_df["poster_gender"].isin(["male", "female"])]

    if len(known_df) == 0:
        df = pd.DataFrame(columns=["Direction", "Male Count", "Female Count", "Male %", "Female %"])
        df.to_csv(TABLE_DIR / "advice_direction.csv", index=False)
        return df

    crosstab = pd.crosstab(known_df["advice_direction"], known_df["poster_gender"])

    male_total = crosstab["male"].sum() if "male" in crosstab.columns else 0
    female_total = crosstab["female"].sum() if "female" in crosstab.columns else 0

    rows = []
    for direction in crosstab.index:
        mc = crosstab.loc[direction, "male"] if "male" in crosstab.columns else 0
        fc = crosstab.loc[direction, "female"] if "female" in crosstab.columns else 0
        rows.append({
            "Direction": direction.replace("_", " ").title(),
            "Male Count": mc,
            "Female Count": fc,
            "Male %": f"{mc/male_total*100:.1f}%" if male_total > 0 else "0%",
            "Female %": f"{fc/female_total*100:.1f}%" if female_total > 0 else "0%"
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "advice_direction.csv", index=False)
    logger.info(f"Saved advice_direction.csv")

    return df


# =============================================================================
# CHART GENERATION FUNCTIONS
# =============================================================================

def fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_gender_distribution(post_df: pd.DataFrame) -> plt.Figure:
    """Generate C1: Gender distribution chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    gender_counts = post_df["poster_gender"].value_counts()
    colors = [COLORS.get(g, "#7A7A7A") for g in gender_counts.index]

    axes[0].pie(gender_counts.values, labels=[g.capitalize() for g in gender_counts.index],
                colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0].set_title("Post Distribution by Gender", fontsize=14)

    # Bar chart
    axes[1].bar(range(len(gender_counts)), gender_counts.values, color=colors)
    axes[1].set_xticks(range(len(gender_counts)))
    axes[1].set_xticklabels([g.capitalize() for g in gender_counts.index])
    axes[1].set_ylabel("Number of Posts")
    axes[1].set_title("Post Count by Gender", fontsize=14)

    # Add value labels
    for i, v in enumerate(gender_counts.values):
        axes[1].text(i, v + 0.5, str(v), ha='center')

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "gender_distribution.png", dpi=150, bbox_inches='tight')
    logger.info("Saved gender_distribution.png")

    return fig


def plot_severity_by_gender(post_df: pd.DataFrame) -> plt.Figure:
    """Generate C2: Severity by gender stacked bar chart."""
    known_df = post_df[post_df["poster_gender"].isin(["male", "female"])]

    if len(known_df) == 0 or known_df["situation_severity"].isna().all():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', fontsize=14)
        ax.set_title("Situation Severity by Gender")
        return fig

    crosstab = pd.crosstab(known_df["poster_gender"], known_df["situation_severity"], normalize='index')

    fig, ax = plt.subplots(figsize=(8, 6))

    severity_order = ["low", "medium", "high"]
    bottom = np.zeros(len(crosstab.index))

    for severity in severity_order:
        if severity in crosstab.columns:
            values = crosstab[severity].values
            ax.bar(crosstab.index, values, bottom=bottom,
                   label=severity.capitalize(), color=SEVERITY_COLORS[severity])
            bottom += values

    ax.set_ylabel("Proportion")
    ax.set_title("Situation Severity by Poster Gender", fontsize=14)
    ax.legend(title="Severity")
    ax.set_xticklabels([g.capitalize() for g in crosstab.index])

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "severity_by_gender.png", dpi=150, bbox_inches='tight')
    logger.info("Saved severity_by_gender.png")

    return fig


def plot_fault_by_gender(post_df: pd.DataFrame) -> plt.Figure:
    """Generate C3: Fault by gender stacked bar chart."""
    known_df = post_df[post_df["poster_gender"].isin(["male", "female"])]

    if len(known_df) == 0 or known_df["op_fault"].isna().all():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', fontsize=14)
        ax.set_title("OP Fault by Gender")
        return fig

    crosstab = pd.crosstab(known_df["poster_gender"], known_df["op_fault"], normalize='index')

    fig, ax = plt.subplots(figsize=(8, 6))

    fault_order = ["none", "some", "substantial", "unclear"]
    bottom = np.zeros(len(crosstab.index))

    for fault in fault_order:
        if fault in crosstab.columns:
            values = crosstab[fault].values
            ax.bar(crosstab.index, values, bottom=bottom,
                   label=fault.capitalize(), color=FAULT_COLORS[fault])
            bottom += values

    ax.set_ylabel("Proportion")
    ax.set_title("OP Fault Level by Poster Gender", fontsize=14)
    ax.legend(title="Fault Level")
    ax.set_xticklabels([g.capitalize() for g in crosstab.index])

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "fault_by_gender.png", dpi=150, bbox_inches='tight')
    logger.info("Saved fault_by_gender.png")

    return fig


def plot_problem_categories(post_df: pd.DataFrame) -> plt.Figure:
    """Generate C4: Problem categories horizontal bar chart."""
    category_counts = post_df["problem_category"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(category_counts))
    ax.barh(y_pos, category_counts.values, color='#3498db')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.replace("_", " ").title() for c in category_counts.index])
    ax.set_xlabel("Number of Posts")
    ax.set_title("Problem Category Distribution", fontsize=14)

    # Add value labels
    for i, v in enumerate(category_counts.values):
        ax.text(v + 0.3, i, str(v), va='center')

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "problem_categories.png", dpi=150, bbox_inches='tight')
    logger.info("Saved problem_categories.png")

    return fig


def plot_tone_heatmap(comment_df: pd.DataFrame) -> plt.Figure:
    """Generate C5: Tone label heatmap by gender."""
    if len(comment_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No comment data available", ha='center', va='center', fontsize=14)
        ax.set_title("Tone Labels by Gender")
        return fig

    # Count tones by gender
    male_counts = defaultdict(int)
    female_counts = defaultdict(int)

    for _, row in comment_df.iterrows():
        for tone in row["tone_labels"]:
            if row["poster_gender"] == "male":
                male_counts[tone] += 1
            elif row["poster_gender"] == "female":
                female_counts[tone] += 1

    all_tones = sorted(set(male_counts.keys()) | set(female_counts.keys()))

    if not all_tones:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No tone data available", ha='center', va='center', fontsize=14)
        return fig

    # Create matrix
    data = np.array([
        [male_counts[t] for t in all_tones],
        [female_counts[t] for t in all_tones]
    ])

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(data, aspect='auto', cmap='YlOrRd')

    ax.set_xticks(np.arange(len(all_tones)))
    ax.set_xticklabels(all_tones, rotation=45, ha='right')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Male', 'Female'])
    ax.set_title("Tone Label Frequency Heatmap", fontsize=14)

    # Add text annotations
    for i in range(2):
        for j in range(len(all_tones)):
            ax.text(j, i, str(data[i, j]), ha='center', va='center', color='black')

    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "tone_heatmap.png", dpi=150, bbox_inches='tight')
    logger.info("Saved tone_heatmap.png")

    return fig


def plot_advice_direction(comment_df: pd.DataFrame) -> plt.Figure:
    """Generate C6: Advice direction grouped bar chart."""
    known_df = comment_df[comment_df["poster_gender"].isin(["male", "female"])]

    if len(known_df) == 0 or known_df["advice_direction"].isna().all():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', fontsize=14)
        ax.set_title("Advice Direction by Gender")
        return fig

    crosstab = pd.crosstab(known_df["advice_direction"], known_df["poster_gender"])

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(crosstab.index))
    width = 0.35

    if "male" in crosstab.columns:
        ax.bar(x - width/2, crosstab["male"], width, label='Male', color=COLORS["male"])
    if "female" in crosstab.columns:
        ax.bar(x + width/2, crosstab["female"], width, label='Female', color=COLORS["female"])

    ax.set_ylabel("Count")
    ax.set_title("Advice Direction by Poster Gender", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", " ").title() for d in crosstab.index])
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "advice_direction.png", dpi=150, bbox_inches='tight')
    logger.info("Saved advice_direction.png")

    return fig


# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def generate_html_report(tables: Dict[str, pd.DataFrame], figures: Dict[str, plt.Figure]) -> str:
    """Generate standalone HTML report."""

    # Convert figures to base64
    figure_data = {}
    for name, fig in figures.items():
        if fig is not None:
            figure_data[name] = fig_to_base64(fig)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gender Bias Research Report</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .figure {{
            text-align: center;
            margin: 20px 0;
        }}
        .figure img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .interpretation {{
            background: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
            font-style: italic;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>Gender Bias in Relationship Advice</h1>
    <p class="timestamp">Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <div class="card">
        <h2>1. Dataset Summary</h2>
        {tables.get('dataset_summary', pd.DataFrame()).to_html(index=False, classes='table')}
    </div>

    <div class="card">
        <h2>2. Gender Distribution</h2>
        {tables.get('gender_distribution', pd.DataFrame()).to_html(index=False, classes='table')}

        <div class="figure">
            <img src="data:image/png;base64,{figure_data.get('gender_distribution', '')}" alt="Gender Distribution">
        </div>

        <div class="interpretation">
            <strong>Interpretation:</strong> This shows the distribution of poster genders in our dataset.
            A high "unknown" count indicates posts where gender could not be determined from the content.
        </div>
    </div>

    <div class="card">
        <h2>3. Confound Balance Analysis</h2>
        <p>Chi-square tests to check if confounding variables are distributed differently by gender:</p>
        {tables.get('confound_balance', pd.DataFrame()).to_html(index=False, classes='table')}

        <div class="interpretation">
            <strong>Interpretation:</strong> A significant p-value (p < 0.05) indicates that the confound
            is distributed differently across genders, which could affect our analysis. Non-significant
            results suggest confounds are balanced.
        </div>
    </div>

    <div class="card">
        <h2>4. Situation Severity by Gender</h2>
        <div class="figure">
            <img src="data:image/png;base64,{figure_data.get('severity_by_gender', '')}" alt="Severity by Gender">
        </div>

        <div class="interpretation">
            <strong>Interpretation:</strong> This shows whether men and women describe situations of
            similar severity, or if one gender tends to describe more severe situations.
        </div>
    </div>

    <div class="card">
        <h2>5. OP Fault by Gender</h2>
        <div class="figure">
            <img src="data:image/png;base64,{figure_data.get('fault_by_gender', '')}" alt="Fault by Gender">
        </div>

        <div class="interpretation">
            <strong>Interpretation:</strong> This shows whether the poster (OP) is classified as being
            at fault differently by gender. If men are more often at fault, harsher responses might
            be justified rather than biased.
        </div>
    </div>

    <div class="card">
        <h2>6. Problem Categories</h2>
        <div class="figure">
            <img src="data:image/png;base64,{figure_data.get('problem_categories', '')}" alt="Problem Categories">
        </div>
    </div>

    <div class="card">
        <h2>7. Comment Tone Analysis</h2>
        <h3>Tone Label Frequencies</h3>
        {tables.get('tone_frequencies', pd.DataFrame()).to_html(index=False, classes='table')}

        <div class="figure">
            <img src="data:image/png;base64,{figure_data.get('tone_heatmap', '')}" alt="Tone Heatmap">
        </div>
    </div>

    <div class="card">
        <h2>8. Advice Direction by Gender</h2>
        {tables.get('advice_direction', pd.DataFrame()).to_html(index=False, classes='table')}

        <div class="figure">
            <img src="data:image/png;base64,{figure_data.get('advice_direction', '')}" alt="Advice Direction">
        </div>

        <div class="interpretation">
            <strong>Interpretation:</strong> This shows whether commenters are more supportive or critical
            based on the poster's gender. A higher proportion of "critical" responses to one gender
            could indicate bias.
        </div>
    </div>

    <div class="card">
        <h2>Methodology Notes</h2>
        <ul>
            <li>Post classifications (gender, severity, fault, category) generated by Claude Haiku</li>
            <li>Comment classifications (tone labels, advice direction) generated by Claude Haiku</li>
            <li>Chi-square tests used to assess confound balance</li>
            <li>Full harshness analysis requires pairwise comparisons (not yet completed)</li>
        </ul>
    </div>
</body>
</html>"""

    # Save HTML report
    report_path = OUTPUT_DIR / "report.html"
    with open(report_path, "w") as f:
        f.write(html)
    logger.info(f"Saved HTML report to {report_path}")

    return html


# =============================================================================
# MAIN REPORT GENERATION
# =============================================================================

def generate_full_report() -> Dict[str, Any]:
    """Generate complete analysis report with all tables and charts."""
    logger.info("=" * 60)
    logger.info("GENERATING ANALYSIS REPORT")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading data...")
    post_df = load_post_data()
    comment_df = load_comment_data()

    logger.info(f"Loaded {len(post_df)} posts, {len(comment_df)} classified comments")

    # Generate tables
    logger.info("\nGenerating tables...")
    tables = {
        "dataset_summary": generate_dataset_summary(),
        "gender_distribution": generate_gender_table(),
        "confound_balance": generate_confound_balance_table(),
        "tone_frequencies": generate_tone_frequency_table(),
        "advice_direction": generate_advice_direction_table(),
    }

    # Generate charts
    logger.info("\nGenerating charts...")
    figures = {
        "gender_distribution": plot_gender_distribution(post_df),
        "severity_by_gender": plot_severity_by_gender(post_df),
        "fault_by_gender": plot_fault_by_gender(post_df),
        "problem_categories": plot_problem_categories(post_df),
        "tone_heatmap": plot_tone_heatmap(comment_df),
        "advice_direction": plot_advice_direction(comment_df),
    }

    # Generate HTML report
    logger.info("\nGenerating HTML report...")
    generate_html_report(tables, figures)

    # Close all figures
    for fig in figures.values():
        if fig is not None:
            plt.close(fig)

    # Print summary to terminal
    print("\n" + "=" * 60)
    print("ANALYSIS REPORT SUMMARY")
    print("=" * 60)

    print("\n--- Dataset Summary ---")
    print(tables["dataset_summary"].to_string(index=False))

    print("\n--- Gender Distribution ---")
    print(tables["gender_distribution"].to_string(index=False))

    print("\n--- Confound Balance (Chi-square tests) ---")
    if len(tables["confound_balance"]) > 0:
        print(tables["confound_balance"].to_string(index=False))
    else:
        print("Insufficient data for chi-square tests")

    print("\n--- Tone Frequencies ---")
    if len(tables["tone_frequencies"]) > 0:
        print(tables["tone_frequencies"].to_string(index=False))
    else:
        print("No tone data available")

    print("\n--- Advice Direction ---")
    if len(tables["advice_direction"]) > 0:
        print(tables["advice_direction"].to_string(index=False))
    else:
        print("No advice direction data available")

    print("\n" + "=" * 60)
    print(f"Tables saved to: {TABLE_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"HTML report saved to: {OUTPUT_DIR / 'report.html'}")
    print("=" * 60)

    return {
        "tables": {k: v.to_dict() for k, v in tables.items()},
        "figures_saved": list(FIGURE_DIR.glob("*.png")),
        "report_path": str(OUTPUT_DIR / "report.html")
    }


if __name__ == "__main__":
    generate_full_report()
