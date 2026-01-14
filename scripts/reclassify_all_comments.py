#!/usr/bin/env python3
"""
Re-classify ALL comments with the updated (more conservative) prompt.

This script:
1. Clears existing comment classifications
2. Re-runs classification on all comments using the updated prompt
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifiers.comment_classifier import CommentClassifier, classify_comments
from src.utils.database import db

DB_PATH = Path(__file__).parent.parent / "data" / "research.db"


def clear_comment_classifications():
    """Delete all existing comment classifications."""
    conn = sqlite3.connect(DB_PATH)

    # Count existing classifications
    count = conn.execute("SELECT COUNT(*) FROM comment_classifications").fetchone()[0]
    print(f"Deleting {count} existing comment classifications...")

    conn.execute("DELETE FROM comment_classifications")
    conn.commit()
    conn.close()

    print("Done. All comment classifications cleared.")
    return count


def main():
    print("=" * 70)
    print("RE-CLASSIFY ALL COMMENTS WITH CONSERVATIVE PROMPT")
    print("=" * 70)

    # Step 1: Clear existing classifications
    print("\nStep 1: Clearing existing classifications...")
    cleared = clear_comment_classifications()

    # Step 2: Re-classify all comments
    print("\nStep 2: Re-classifying all comments...")
    print("This will take a while (est. 1-2 hours for ~7000 comments)...\n")

    results = classify_comments(batch_size=50000)

    print("\n" + "=" * 70)
    print("RE-CLASSIFICATION COMPLETE")
    print("=" * 70)
    print(f"Cleared:    {cleared} old classifications")
    print(f"Classified: {results.get('classified', 0)} comments")
    print(f"Advice:     {results.get('advice_comments', 0)} comments")
    print(f"Errors:     {results.get('errors', 0)}")

    if 'direction_counts' in results:
        print("\nAdvice direction breakdown:")
        for direction, count in results['direction_counts'].items():
            print(f"  {direction}: {count}")


if __name__ == "__main__":
    main()
