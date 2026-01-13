#!/usr/bin/env python3
"""
Compare local LLM (Ollama) classification vs Haiku for comment tone analysis.
"""

import json
import sqlite3
import requests
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "research.db"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

PROMPT_TEMPLATE = """Analyze this comment on a relationship advice post and classify its tone.

POST CONTEXT:
{post_body}

COMMENT TO ANALYZE:
{comment_body}

Respond with ONLY valid JSON in this exact format:
{{
    "tone_labels": ["label1", "label2"],
    "advice_direction": "supportive_of_op|critical_of_op|neutral|mixed"
}}

Valid tone_labels: supportive, critical, empathetic, blunt, constructive, dismissive, encouraging, judgmental, gentle, harsh, sarcastic, understanding, practical

JSON response:"""


def get_classified_comments():
    """Get comments that have been classified by Haiku."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT c.comment_id as id, c.body, cc.tone_labels, cc.advice_direction,
               p.body as post_body
        FROM comments c
        JOIN comment_classifications cc ON c.comment_id = cc.comment_id
        JOIN posts p ON c.post_id = p.post_id
        WHERE cc.tone_labels IS NOT NULL
        LIMIT 10
    """)

    comments = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return comments


def classify_with_ollama(post_body: str, comment_body: str) -> dict:
    """Classify comment using local Ollama model."""
    prompt = PROMPT_TEMPLATE.format(
        post_body=post_body[:2000],  # Truncate for context
        comment_body=comment_body
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        },
        timeout=120
    )

    if response.status_code != 200:
        return {"error": f"Ollama error: {response.status_code}"}

    result = response.json()
    text = result.get("response", "")

    # Try to parse JSON from response
    try:
        # Find JSON in response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass

    return {"error": "Failed to parse JSON", "raw": text[:200]}


def compare_results(haiku: dict, local: dict) -> dict:
    """Compare Haiku vs local model results."""
    haiku_tones = set(json.loads(haiku["tone_labels"]) if haiku["tone_labels"] else [])
    local_tones = set(local.get("tone_labels", []))

    haiku_dir = haiku["advice_direction"]
    local_dir = local.get("advice_direction", "")

    tone_overlap = len(haiku_tones & local_tones) / max(len(haiku_tones | local_tones), 1)
    direction_match = haiku_dir == local_dir

    return {
        "haiku_tones": sorted(haiku_tones),
        "local_tones": sorted(local_tones),
        "tone_jaccard": round(tone_overlap, 2),
        "haiku_direction": haiku_dir,
        "local_direction": local_dir,
        "direction_match": direction_match
    }


def main():
    print(f"Comparing {MODEL} vs Claude Haiku on classified comments\n")
    print("=" * 70)

    comments = get_classified_comments()
    print(f"Testing on {len(comments)} comments\n")

    results = []

    for i, comment in enumerate(comments, 1):
        print(f"[{i}/{len(comments)}] Comment {comment['id'][:8]}...")

        local_result = classify_with_ollama(comment["post_body"], comment["body"])

        if "error" in local_result:
            print(f"  ERROR: {local_result['error']}")
            continue

        comparison = compare_results(comment, local_result)
        results.append(comparison)

        print(f"  Haiku tones:  {comparison['haiku_tones']}")
        print(f"  Local tones:  {comparison['local_tones']}")
        print(f"  Tone overlap: {comparison['tone_jaccard']:.0%}")
        print(f"  Direction:    Haiku={comparison['haiku_direction']}, Local={comparison['local_direction']} {'✓' if comparison['direction_match'] else '✗'}")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        avg_overlap = sum(r["tone_jaccard"] for r in results) / len(results)
        direction_matches = sum(1 for r in results if r["direction_match"])

        print(f"Average tone overlap (Jaccard): {avg_overlap:.1%}")
        print(f"Direction agreement: {direction_matches}/{len(results)} ({direction_matches/len(results):.1%})")

        if avg_overlap < 0.5:
            print("\n⚠️  Low agreement - local model may not be suitable for this task")
        elif avg_overlap < 0.7:
            print("\n⚡ Moderate agreement - local model usable for rough filtering")
        else:
            print("\n✓ Good agreement - local model viable for this task")


if __name__ == "__main__":
    main()
