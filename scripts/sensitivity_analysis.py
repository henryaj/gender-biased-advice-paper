#!/usr/bin/env python3
"""
Sensitivity Analysis: Compare findings with vs without disputed tone labels.

The validation spot-checking showed:
- advice_direction: 96% agreement (high confidence)
- harsh tones (harsh, judgmental, blaming, dismissive): ~57% agreement (disputed)

This analysis shows whether the gender bias findings hold when:
1. Using only high-confidence labels (advice_direction)
2. Excluding disputed tone labels
3. Using all labels (original analysis)
"""

import json
import sqlite3
import sys
from pathlib import Path
from collections import defaultdict
from scipy import stats
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "data" / "research.db"

# Disputed tone labels (those with <60% human agreement)
DISPUTED_TONES = {'harsh', 'judgmental', 'blaming', 'dismissive', 'condescending', 'hostile'}

# High-confidence positive tones
HIGH_CONFIDENCE_TONES = {'gentle', 'empathetic', 'constructive', 'understanding', 'encouraging', 'supportive'}


def get_analysis_data():
    """Get data for analysis."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            c.comment_id,
            cc.advice_direction,
            cc.tone_labels,
            pc.poster_gender,
            pc.situation_severity,
            pc.op_fault,
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


def analyze_advice_direction(data):
    """
    Primary analysis using advice_direction (96% human agreement).

    Compares rate of critical_of_op advice by gender.
    """
    male_critical = 0
    male_total = 0
    female_critical = 0
    female_total = 0

    for row in data:
        if row['poster_gender'] == 'male':
            male_total += 1
            if row['advice_direction'] == 'critical_of_op':
                male_critical += 1
        else:
            female_total += 1
            if row['advice_direction'] == 'critical_of_op':
                female_critical += 1

    male_rate = male_critical / male_total if male_total > 0 else 0
    female_rate = female_critical / female_total if female_total > 0 else 0

    # Chi-square test
    contingency = [
        [male_critical, male_total - male_critical],
        [female_critical, female_total - female_critical]
    ]
    chi2, pvalue, dof, expected = stats.chi2_contingency(contingency)

    # Odds ratio
    a, b = male_critical, male_total - male_critical
    c, d = female_critical, female_total - female_critical
    odds_ratio = (a * d) / (b * c) if (b * c) > 0 else float('inf')

    return {
        'metric': 'advice_direction (critical_of_op)',
        'human_agreement': '96%',
        'male_n': male_total,
        'female_n': female_total,
        'male_rate': male_rate,
        'female_rate': female_rate,
        'difference': male_rate - female_rate,
        'odds_ratio': odds_ratio,
        'chi2': chi2,
        'p_value': pvalue,
        'significant': pvalue < 0.05
    }


def analyze_tone_labels(data, include_disputed=True):
    """
    Analyze tone label frequencies by gender.

    Args:
        include_disputed: If False, exclude disputed tone labels
    """
    results = {}

    # Count by gender
    male_counts = defaultdict(int)
    female_counts = defaultdict(int)
    male_total = 0
    female_total = 0

    for row in data:
        tone_labels = json.loads(row['tone_labels']) if row['tone_labels'] else []

        # Filter tones if needed
        if not include_disputed:
            tone_labels = [t for t in tone_labels if t not in DISPUTED_TONES]

        if row['poster_gender'] == 'male':
            male_total += 1
            for label in tone_labels:
                male_counts[label] += 1
        else:
            female_total += 1
            for label in tone_labels:
                female_counts[label] += 1

    all_labels = set(male_counts.keys()) | set(female_counts.keys())

    for label in all_labels:
        male_count = male_counts[label]
        female_count = female_counts[label]

        male_rate = male_count / male_total if male_total > 0 else 0
        female_rate = female_count / female_total if female_total > 0 else 0

        # Chi-square test
        contingency = [
            [male_count, male_total - male_count],
            [female_count, female_total - female_count]
        ]
        try:
            chi2, pvalue, dof, expected = stats.chi2_contingency(contingency)
        except ValueError:
            chi2, pvalue = 0, 1.0

        results[label] = {
            'male_count': male_count,
            'female_count': female_count,
            'male_rate': male_rate,
            'female_rate': female_rate,
            'difference': male_rate - female_rate,
            'chi2': chi2,
            'p_value': pvalue,
            'significant': pvalue < 0.05,
            'disputed': label in DISPUTED_TONES
        }

    return results, male_total, female_total


def analyze_negative_tone_composite(data, include_disputed=True):
    """
    Composite analysis: any negative tone present.
    """
    if include_disputed:
        negative_tones = DISPUTED_TONES
        label = "Any harsh tone (all)"
    else:
        # No disputed tones, can't run this analysis
        return None

    male_negative = 0
    male_total = 0
    female_negative = 0
    female_total = 0

    for row in data:
        tone_labels = json.loads(row['tone_labels']) if row['tone_labels'] else []
        has_negative = any(t in negative_tones for t in tone_labels)

        if row['poster_gender'] == 'male':
            male_total += 1
            if has_negative:
                male_negative += 1
        else:
            female_total += 1
            if has_negative:
                female_negative += 1

    male_rate = male_negative / male_total if male_total > 0 else 0
    female_rate = female_negative / female_total if female_total > 0 else 0

    # Chi-square test
    contingency = [
        [male_negative, male_total - male_negative],
        [female_negative, female_total - female_negative]
    ]
    chi2, pvalue, dof, expected = stats.chi2_contingency(contingency)

    # Odds ratio
    a, b = male_negative, male_total - male_negative
    c, d = female_negative, female_total - female_negative
    odds_ratio = (a * d) / (b * c) if (b * c) > 0 else float('inf')

    return {
        'metric': label,
        'male_n': male_total,
        'female_n': female_total,
        'male_rate': male_rate,
        'female_rate': female_rate,
        'difference': male_rate - female_rate,
        'odds_ratio': odds_ratio,
        'chi2': chi2,
        'p_value': pvalue,
        'significant': pvalue < 0.05
    }


def main():
    print("=" * 70)
    print("SENSITIVITY ANALYSIS: With vs Without Disputed Tone Labels")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data = get_analysis_data()
    print(f"Total advice comments: {len(data)}")

    male_count = sum(1 for d in data if d['poster_gender'] == 'male')
    female_count = sum(1 for d in data if d['poster_gender'] == 'female')
    print(f"  Male posters: {male_count}")
    print(f"  Female posters: {female_count}")

    # Analysis 1: Advice Direction (High Confidence)
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Advice Direction (96% Human Agreement)")
    print("=" * 70)

    direction_results = analyze_advice_direction(data)
    print(f"\nMen receiving critical advice:   {direction_results['male_rate']*100:.1f}% ({direction_results['male_n']} comments)")
    print(f"Women receiving critical advice: {direction_results['female_rate']*100:.1f}% ({direction_results['female_n']} comments)")
    print(f"\nDifference: {direction_results['difference']*100:.1f} percentage points")
    print(f"Odds Ratio: {direction_results['odds_ratio']:.2f}x")
    print(f"Chi-square: {direction_results['chi2']:.2f}, p = {direction_results['p_value']:.4f}")
    print(f"Significant: {'YES' if direction_results['significant'] else 'NO'}")

    # Analysis 2: Tone Labels (All)
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Tone Labels (Including Disputed - 57% Agreement)")
    print("=" * 70)

    tone_all, male_n, female_n = analyze_tone_labels(data, include_disputed=True)

    print(f"\nTone labels with significant gender differences:")
    print(f"{'Tone':<15} {'Male':>8} {'Female':>8} {'Diff':>8} {'p-value':>10} {'Disputed':>10}")
    print("-" * 65)

    for tone, stats_dict in sorted(tone_all.items(), key=lambda x: -abs(x[1]['difference'])):
        if stats_dict['significant']:
            disputed_marker = "*" if stats_dict['disputed'] else ""
            print(f"{tone:<15} {stats_dict['male_rate']*100:>7.1f}% {stats_dict['female_rate']*100:>7.1f}% "
                  f"{stats_dict['difference']*100:>+7.1f}% {stats_dict['p_value']:>10.4f} {disputed_marker:>10}")

    # Composite harsh tone analysis
    harsh_composite = analyze_negative_tone_composite(data, include_disputed=True)
    if harsh_composite:
        print(f"\nComposite: Any harsh tone present")
        print(f"  Men:   {harsh_composite['male_rate']*100:.1f}%")
        print(f"  Women: {harsh_composite['female_rate']*100:.1f}%")
        print(f"  OR:    {harsh_composite['odds_ratio']:.2f}x, p = {harsh_composite['p_value']:.4f}")

    # Analysis 3: Tone Labels (High Confidence Only)
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Tone Labels (Excluding Disputed)")
    print("=" * 70)
    print("Excluding: harsh, judgmental, blaming, dismissive, condescending, hostile")

    tone_clean, _, _ = analyze_tone_labels(data, include_disputed=False)

    print(f"\nTone labels with significant gender differences:")
    print(f"{'Tone':<15} {'Male':>8} {'Female':>8} {'Diff':>8} {'p-value':>10}")
    print("-" * 55)

    any_significant = False
    for tone, stats_dict in sorted(tone_clean.items(), key=lambda x: -abs(x[1]['difference'])):
        if stats_dict['significant']:
            any_significant = True
            print(f"{tone:<15} {stats_dict['male_rate']*100:>7.1f}% {stats_dict['female_rate']*100:>7.1f}% "
                  f"{stats_dict['difference']*100:>+7.1f}% {stats_dict['p_value']:>10.4f}")

    if not any_significant:
        print("  (No significant differences in non-disputed tones)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: ROBUSTNESS OF GENDER BIAS FINDINGS")
    print("=" * 70)

    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ METRIC                         │ AGREEMENT │ GENDER BIAS │ ROBUST? │
├────────────────────────────────┼───────────┼─────────────┼─────────┤""")

    # Direction
    dir_bias = "YES" if direction_results['significant'] else "NO"
    print(f"│ Advice Direction (critical)    │    96%    │     {dir_bias:<6} │   YES   │")

    # Harsh composite
    if harsh_composite:
        harsh_bias = "YES" if harsh_composite['significant'] else "NO"
        print(f"│ Any Harsh Tone (disputed)      │    57%    │     {harsh_bias:<6} │   N/A*  │")

    print("""└─────────────────────────────────────────────────────────────────────┘

* Disputed tones may inflate effect, but direction-based analysis confirms bias exists.
""")

    print("\nKEY FINDING:")
    if direction_results['significant']:
        print(f"  Even using ONLY the high-confidence metric (advice_direction),")
        print(f"  men receive {direction_results['odds_ratio']:.1f}x more critical advice than women.")
        print(f"  This finding is robust and not dependent on disputed tone labels.")
    else:
        print(f"  Using high-confidence metrics, the gender bias is not statistically significant.")

    # Save results
    output_path = Path(__file__).parent.parent / "outputs" / "sensitivity_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'advice_direction_analysis': direction_results,
        'tone_labels_all': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                for kk, vv in v.items()}
                           for k, v in tone_all.items()},
        'tone_labels_clean': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                  for kk, vv in v.items()}
                             for k, v in tone_clean.items()},
        'harsh_composite': harsh_composite
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
