#!/usr/bin/env python3
"""Generate charts for the gender bias paper."""

import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "research.db"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Style settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

MALE_COLOR = '#4A90D9'
FEMALE_COLOR = '#E85D75'


def get_data():
    """Get analysis data from database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            cc.advice_direction,
            pc.poster_gender,
            pc.problem_category
        FROM comments c
        JOIN comment_classifications cc ON c.comment_id = cc.comment_id
        JOIN posts p ON c.post_id = p.post_id
        JOIN post_classifications pc ON p.post_id = pc.post_id
        WHERE cc.is_advice = 1
        AND pc.is_relationship_advice = 1
        AND pc.poster_gender IN ('male', 'female')
    """

    rows = conn.execute(query).fetchall()
    conn.close()

    return [dict(row) for row in rows]


def chart_critical_by_gender(data):
    """Bar chart: Critical advice rate by gender."""
    male_total = sum(1 for d in data if d['poster_gender'] == 'male')
    female_total = sum(1 for d in data if d['poster_gender'] == 'female')
    male_critical = sum(1 for d in data if d['poster_gender'] == 'male' and d['advice_direction'] == 'critical_of_op')
    female_critical = sum(1 for d in data if d['poster_gender'] == 'female' and d['advice_direction'] == 'critical_of_op')

    male_rate = male_critical / male_total * 100
    female_rate = female_critical / female_total * 100

    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(['Men', 'Women'], [male_rate, female_rate],
                  color=[MALE_COLOR, FEMALE_COLOR], width=0.6)

    # Add value labels
    for bar, rate in zip(bars, [male_rate, female_rate]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Critical Advice Rate (%)')
    ax.set_title('Men Receive 4x More Critical Advice', fontweight='bold', pad=15)
    ax.set_ylim(0, 35)

    # Add odds ratio annotation
    ax.annotate('OR = 4.19x\np < 0.0001',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'chart_critical_by_gender.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'chart_critical_by_gender.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: chart_critical_by_gender.png/pdf")


def chart_by_category(data):
    """Horizontal bar chart: Critical rate by problem category."""
    categories = {}

    for row in data:
        cat = row['problem_category'] or 'unknown'
        if cat not in categories:
            categories[cat] = {'male_critical': 0, 'male_total': 0, 'female_critical': 0, 'female_total': 0}

        if row['poster_gender'] == 'male':
            categories[cat]['male_total'] += 1
            if row['advice_direction'] == 'critical_of_op':
                categories[cat]['male_critical'] += 1
        else:
            categories[cat]['female_total'] += 1
            if row['advice_direction'] == 'critical_of_op':
                categories[cat]['female_critical'] += 1

    # Filter to categories with enough data
    valid_cats = {k: v for k, v in categories.items()
                  if v['male_total'] >= 20 and v['female_total'] >= 20}

    # Calculate rates and sort by difference
    cat_data = []
    for cat, counts in valid_cats.items():
        male_rate = counts['male_critical'] / counts['male_total'] * 100
        female_rate = counts['female_critical'] / counts['female_total'] * 100
        cat_data.append((cat.capitalize(), male_rate, female_rate, male_rate - female_rate))

    cat_data.sort(key=lambda x: x[3], reverse=True)

    labels = [d[0] for d in cat_data]
    male_rates = [d[1] for d in cat_data]
    female_rates = [d[2] for d in cat_data]

    fig, ax = plt.subplots(figsize=(8, 5))

    y = np.arange(len(labels))
    height = 0.35

    bars1 = ax.barh(y - height/2, male_rates, height, label='Men', color=MALE_COLOR)
    bars2 = ax.barh(y + height/2, female_rates, height, label='Women', color=FEMALE_COLOR)

    ax.set_xlabel('Critical Advice Rate (%)')
    ax.set_title('Gender Gap Across Problem Categories', fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.legend(loc='lower right')
    ax.set_xlim(0, 55)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'chart_by_category.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'chart_by_category.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: chart_by_category.png/pdf")


def chart_advice_distribution(data):
    """Stacked bar: All advice directions by gender."""
    directions = ['critical_of_op', 'mixed', 'neutral', 'supportive_of_op']
    labels = ['Critical', 'Mixed', 'Neutral', 'Supportive']
    colors = ['#D32F2F', '#FF9800', '#9E9E9E', '#4CAF50']

    male_counts = {d: 0 for d in directions}
    female_counts = {d: 0 for d in directions}
    male_total = 0
    female_total = 0

    for row in data:
        direction = row['advice_direction']
        if direction in directions:
            if row['poster_gender'] == 'male':
                male_counts[direction] += 1
                male_total += 1
            else:
                female_counts[direction] += 1
                female_total += 1

    male_rates = [male_counts[d] / male_total * 100 for d in directions]
    female_rates = [female_counts[d] / female_total * 100 for d in directions]

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.array([0, 1])
    width = 0.5

    bottom_male = 0
    bottom_female = 0

    for i, (direction, label, color) in enumerate(zip(directions, labels, colors)):
        male_val = male_rates[i]
        female_val = female_rates[i]

        ax.bar(0, male_val, width, bottom=bottom_male, color=color, label=label if i == 0 else None)
        ax.bar(1, female_val, width, bottom=bottom_female, color=color)

        # Add labels for significant segments
        if male_val > 8:
            ax.text(0, bottom_male + male_val/2, f'{male_val:.0f}%',
                    ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        if female_val > 8:
            ax.text(1, bottom_female + female_val/2, f'{female_val:.0f}%',
                    ha='center', va='center', color='white', fontweight='bold', fontsize=9)

        bottom_male += male_val
        bottom_female += female_val

    ax.set_ylabel('Percentage of Advice')
    ax.set_title('Distribution of Advice Types by Gender', fontweight='bold', pad=15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Men\n(n=1,716)', 'Women\n(n=4,364)'])
    ax.set_ylim(0, 100)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'chart_advice_distribution.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'chart_advice_distribution.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: chart_advice_distribution.png/pdf")


def main():
    print("Generating charts...")
    data = get_data()
    print(f"Loaded {len(data)} comments")

    chart_critical_by_gender(data)
    chart_by_category(data)
    chart_advice_distribution(data)

    print("\nAll charts generated!")


if __name__ == "__main__":
    main()
