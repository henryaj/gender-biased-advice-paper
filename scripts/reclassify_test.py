#!/usr/bin/env python3
"""
Re-classify a test batch of comments with the updated (more conservative) prompt.
Compare old vs new classifications to measure the impact of prompt changes.
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifiers.comment_classifier import CommentClassifier
from src.utils.database import db

HARSH_TONES = ['harsh', 'judgmental', 'blaming', 'dismissive', 'condescending', 'hostile']


def get_comments_with_harsh_tones(limit: int = 100):
    """Get comments that were originally classified with harsh tones."""
    conn = sqlite3.connect(Path(__file__).parent.parent / "data" / "research.db")
    conn.row_factory = sqlite3.Row

    # Get comments with at least one harsh tone label
    query = """
        SELECT c.comment_id, c.body, cc.tone_labels, cc.advice_direction,
               p.title as post_title, pc.brief_situation_summary
        FROM comments c
        JOIN comment_classifications cc ON c.comment_id = cc.comment_id
        JOIN posts p ON c.post_id = p.post_id
        JOIN post_classifications pc ON p.post_id = pc.post_id
        WHERE cc.is_advice = 1
        AND (
            cc.tone_labels LIKE '%"harsh"%'
            OR cc.tone_labels LIKE '%"judgmental"%'
            OR cc.tone_labels LIKE '%"blaming"%'
            OR cc.tone_labels LIKE '%"dismissive"%'
            OR cc.tone_labels LIKE '%"condescending"%'
            OR cc.tone_labels LIKE '%"hostile"%'
        )
        ORDER BY RANDOM()
        LIMIT ?
    """

    rows = conn.execute(query, (limit,)).fetchall()
    conn.close()

    return [dict(row) for row in rows]


def count_harsh_tones(tone_labels: list) -> int:
    """Count how many harsh tone labels are present."""
    return sum(1 for t in tone_labels if t in HARSH_TONES)


def main():
    print("=" * 70)
    print("RE-CLASSIFICATION TEST WITH CONSERVATIVE PROMPT")
    print("=" * 70)

    # Get test batch
    print("\nFetching comments with harsh tones...")
    comments = get_comments_with_harsh_tones(limit=100)
    print(f"Found {len(comments)} comments to re-classify\n")

    # Initialize classifier (uses updated prompt)
    classifier = CommentClassifier()

    # Track results
    results = {
        'total': len(comments),
        'harsh_reduced': 0,
        'harsh_same': 0,
        'harsh_increased': 0,
        'direction_changed': 0,
        'old_harsh_total': 0,
        'new_harsh_total': 0,
        'examples': []
    }

    for i, comment in enumerate(comments, 1):
        old_tones = json.loads(comment['tone_labels'])
        old_direction = comment['advice_direction']
        old_harsh_count = count_harsh_tones(old_tones)
        results['old_harsh_total'] += old_harsh_count

        # Re-classify with new prompt
        post_context = comment.get('brief_situation_summary') or comment.get('post_title', '')
        new_result = classifier.classify_comment(
            comment_body=comment['body'],
            post_context=post_context,
            comment_id=comment['comment_id']
        )

        new_tones = new_result.tone_labels
        new_direction = new_result.advice_direction
        new_harsh_count = count_harsh_tones(new_tones)
        results['new_harsh_total'] += new_harsh_count

        # Compare
        if new_harsh_count < old_harsh_count:
            results['harsh_reduced'] += 1
        elif new_harsh_count > old_harsh_count:
            results['harsh_increased'] += 1
        else:
            results['harsh_same'] += 1

        if new_direction != old_direction:
            results['direction_changed'] += 1

        # Save interesting examples (where harsh labels were removed)
        if new_harsh_count < old_harsh_count and len(results['examples']) < 5:
            removed_tones = set(old_tones) - set(new_tones)
            results['examples'].append({
                'comment': comment['body'][:200] + '...' if len(comment['body']) > 200 else comment['body'],
                'old_tones': old_tones,
                'new_tones': new_tones,
                'removed': list(removed_tones & set(HARSH_TONES))
            })

        # Progress
        if i % 20 == 0:
            print(f"Progress: {i}/{len(comments)}")

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nComments re-classified: {results['total']}")
    print(f"\nHarsh label changes:")
    print(f"  Reduced (fewer harsh): {results['harsh_reduced']} ({results['harsh_reduced']/results['total']*100:.1f}%)")
    print(f"  Same:                  {results['harsh_same']} ({results['harsh_same']/results['total']*100:.1f}%)")
    print(f"  Increased:             {results['harsh_increased']} ({results['harsh_increased']/results['total']*100:.1f}%)")

    print(f"\nTotal harsh labels:")
    print(f"  Old prompt: {results['old_harsh_total']}")
    print(f"  New prompt: {results['new_harsh_total']}")
    reduction = (results['old_harsh_total'] - results['new_harsh_total']) / results['old_harsh_total'] * 100
    print(f"  Reduction:  {reduction:.1f}%")

    print(f"\nAdvice direction changes: {results['direction_changed']} ({results['direction_changed']/results['total']*100:.1f}%)")

    if results['examples']:
        print("\n" + "=" * 70)
        print("EXAMPLE CHANGES")
        print("=" * 70)
        for i, ex in enumerate(results['examples'], 1):
            print(f"\n--- Example {i} ---")
            print(f"Comment: {ex['comment']}")
            print(f"Removed labels: {ex['removed']}")
            print(f"Old tones: {ex['old_tones']}")
            print(f"New tones: {ex['new_tones']}")


if __name__ == "__main__":
    main()
