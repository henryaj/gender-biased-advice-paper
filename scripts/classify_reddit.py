#!/usr/bin/env python3
"""
Reddit Classification Pipeline

For Reddit posts from r/relationship_advice:
1. Gender is already extracted during scraping (stored in flair field)
2. is_relationship_advice is always True (all from relationship subreddit)
3. Still need LLM for: situation_severity, op_fault, problem_category
4. Comments need full LLM classification (same as Metafilter)
"""

import sys
import re
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import db
from src.utils.logging import get_logger

logger = get_logger("classify_reddit")


def extract_gender_from_flair(flair: str) -> tuple[str, float]:
    """
    Extract gender from Reddit post flair.

    Flair format: "gender:male,age:27" or "gender:female,age:30"

    Returns:
        (gender, confidence) tuple
    """
    if not flair:
        return "unknown", 0.0

    match = re.search(r'gender:(\w+)', flair)
    if match:
        gender = match.group(1).lower()
        if gender in ('male', 'female'):
            # High confidence since extracted from explicit [25M] markers
            return gender, 0.95

    return "unknown", 0.0


def create_reddit_post_classifications(dry_run: bool = False) -> dict:
    """
    Create post_classifications for Reddit posts using pre-extracted gender.

    This step populates the basic fields (gender, is_relationship_advice)
    without needing LLM calls. The severity/fault/category fields can be
    filled in by a subsequent LLM classification step.

    Args:
        dry_run: If True, don't actually insert, just report counts

    Returns:
        Dictionary with counts
    """
    logger.info("Creating post classifications for Reddit posts...")

    with db.get_connection() as conn:
        # Get Reddit posts without classifications
        posts = conn.execute("""
            SELECT p.post_id, p.title, p.body, p.flair, p.raw_json
            FROM posts p
            LEFT JOIN post_classifications pc ON p.post_id = pc.post_id
            WHERE p.post_id LIKE 'reddit_%'
            AND pc.post_id IS NULL
        """).fetchall()

        logger.info(f"Found {len(posts)} Reddit posts without classifications")

        if dry_run:
            return {"found": len(posts), "dry_run": True}

        created = 0
        male = 0
        female = 0
        unknown = 0

        for post in posts:
            post_id = post['post_id']
            flair = post['flair'] or ''
            raw_json = json.loads(post['raw_json']) if post['raw_json'] else {}

            # Extract gender from flair
            gender, confidence = extract_gender_from_flair(flair)

            # Try raw_json as fallback (has op_gender directly)
            if gender == "unknown" and raw_json.get('op_gender'):
                gender = raw_json['op_gender']
                confidence = 0.95

            # Insert classification
            conn.execute("""
                INSERT INTO post_classifications (
                    post_id, is_relationship_advice, poster_gender,
                    gender_confidence, relationship_type, brief_situation_summary,
                    classification_model, situation_severity, op_fault, problem_category
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                post_id,
                True,  # All Reddit posts are from r/relationship_advice
                gender,
                confidence,
                "romantic",  # Most are romantic, can refine later
                "",  # Will be filled by LLM if needed
                "extracted_from_title",  # Not LLM-classified yet
                None,  # To be classified
                None,  # To be classified
                None   # To be classified
            ))

            created += 1
            if gender == "male":
                male += 1
            elif gender == "female":
                female += 1
            else:
                unknown += 1

        conn.commit()

    logger.info(f"Created {created} classifications: {male} male, {female} female, {unknown} unknown")

    return {
        "created": created,
        "male": male,
        "female": female,
        "unknown": unknown
    }


def classify_reddit_post_details(
    batch_size: int = 100,
    model: str = "claude-haiku-4-5-20251001"
) -> dict:
    """
    Classify situation_severity, op_fault, and problem_category for Reddit posts.

    This is a simplified classification that only extracts the confound fields,
    since gender is already known.
    """
    import anthropic
    import os

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    logger.info(f"Classifying Reddit post details with {model}...")

    with db.get_connection() as conn:
        # Get Reddit posts that need detail classification
        posts = conn.execute("""
            SELECT p.post_id, p.title, p.body, pc.poster_gender
            FROM posts p
            JOIN post_classifications pc ON p.post_id = pc.post_id
            WHERE p.post_id LIKE 'reddit_%'
            AND pc.situation_severity IS NULL
            LIMIT ?
        """, (batch_size,)).fetchall()

        logger.info(f"Found {len(posts)} Reddit posts needing detail classification")

        classified = 0
        errors = 0

        for post in posts:
            try:
                prompt = f"""Analyze this relationship advice post and extract:

<post>
Title: {post['title']}

Body:
{post['body'][:6000]}
</post>

Return ONLY a JSON object with these fields:

1. "situation_severity": "low" | "medium" | "high"
   - "low": Minor disagreement, miscommunication, preference differences
   - "medium": Concerning behavior, relationship strain, trust issues
   - "high": Abuse, safety concerns, cheating/betrayal, potentially relationship-ending

2. "op_fault": "none" | "some" | "substantial" | "unclear"
   - "none": OP is clearly the wronged party
   - "some": OP contributed but isn't primarily at fault
   - "substantial": OP is primarily at fault
   - "unclear": Ambiguous or genuinely shared fault

3. "problem_category": one of:
   - "communication", "infidelity", "boundaries", "commitment",
   - "intimacy", "finances", "family_inlaws", "abuse_safety",
   - "jealousy_trust", "lifestyle", "other"

4. "situation_summary": Brief (1-2 sentence) neutral summary

{{"situation_severity": "...", "op_fault": "...", "problem_category": "...", "situation_summary": "..."}}"""

                response = client.messages.create(
                    model=model,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}]
                )

                text = response.content[0].text.strip()

                # Parse JSON
                if text.startswith("```"):
                    text = "\n".join(line for line in text.split("\n")
                                    if not line.startswith("```"))

                result = json.loads(text)

                # Update classification
                conn.execute("""
                    UPDATE post_classifications
                    SET situation_severity = ?,
                        op_fault = ?,
                        problem_category = ?,
                        brief_situation_summary = ?,
                        classification_model = ?
                    WHERE post_id = ?
                """, (
                    result.get('situation_severity'),
                    result.get('op_fault'),
                    result.get('problem_category'),
                    result.get('situation_summary', ''),
                    model,
                    post['post_id']
                ))

                classified += 1

                if classified % 10 == 0:
                    conn.commit()
                    logger.info(f"Progress: {classified}/{len(posts)}")

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error classifying {post['post_id']}: {e}")
                errors += 1

        conn.commit()

    logger.info(f"Classified {classified} posts, {errors} errors")
    return {"classified": classified, "errors": errors}


def classify_reddit_comments(
    batch_size: int = 1000,
    model: str = "claude-sonnet-4-5-20250929"
) -> dict:
    """
    Classify comments on Reddit posts using the standard comment classifier.
    """
    from src.classifiers.comment_classifier import CommentClassifier

    classifier = CommentClassifier(model=model)

    logger.info(f"Classifying Reddit comments with {model}...")

    with db.get_connection() as conn:
        # Get unclassified comments on Reddit posts
        comments = conn.execute("""
            SELECT c.comment_id, c.body, c.post_id,
                   p.title as post_title,
                   pc.brief_situation_summary, pc.poster_gender
            FROM comments c
            JOIN posts p ON c.post_id = p.post_id
            JOIN post_classifications pc ON p.post_id = pc.post_id
            LEFT JOIN comment_classifications cc ON c.comment_id = cc.comment_id
            WHERE c.post_id LIKE 'reddit_%'
            AND cc.comment_id IS NULL
            AND pc.poster_gender IN ('male', 'female')
            LIMIT ?
        """, (batch_size,)).fetchall()

        logger.info(f"Found {len(comments)} Reddit comments to classify")

        classified = 0
        advice_count = 0
        errors = 0

        direction_counts = {
            "supportive_of_op": 0,
            "critical_of_op": 0,
            "neutral": 0,
            "mixed": 0
        }

        for comment in comments:
            try:
                post_context = comment['brief_situation_summary'] or f"Title: {comment['post_title']}"

                result = classifier.classify_comment(
                    comment_body=comment['body'],
                    post_context=post_context,
                    comment_id=comment['comment_id']
                )

                db.insert_comment_classification(
                    comment_id=comment['comment_id'],
                    is_advice=result.is_advice,
                    advice_direction=result.advice_direction,
                    tone_labels=result.tone_labels,
                    classification_model=model
                )

                classified += 1
                if result.is_advice:
                    advice_count += 1
                    if result.advice_direction:
                        direction_counts[result.advice_direction] += 1

                if classified % 100 == 0:
                    logger.info(f"Progress: {classified}/{len(comments)}, {advice_count} advice")

                time.sleep(0.05)

            except Exception as e:
                logger.error(f"Error classifying {comment['comment_id']}: {e}")
                errors += 1

    logger.info(f"Classified {classified} comments, {advice_count} advice, {errors} errors")
    logger.info(f"Direction counts: {direction_counts}")

    return {
        "classified": classified,
        "advice_count": advice_count,
        "errors": errors,
        "direction_counts": direction_counts
    }


def get_reddit_stats() -> dict:
    """Get current statistics for Reddit data."""
    with db.get_connection() as conn:
        stats = {}

        # Posts
        stats['total_posts'] = conn.execute(
            "SELECT COUNT(*) FROM posts WHERE post_id LIKE 'reddit_%'"
        ).fetchone()[0]

        stats['classified_posts'] = conn.execute("""
            SELECT COUNT(*) FROM post_classifications pc
            JOIN posts p ON pc.post_id = p.post_id
            WHERE p.post_id LIKE 'reddit_%'
        """).fetchone()[0]

        stats['posts_with_details'] = conn.execute("""
            SELECT COUNT(*) FROM post_classifications pc
            JOIN posts p ON pc.post_id = p.post_id
            WHERE p.post_id LIKE 'reddit_%'
            AND pc.situation_severity IS NOT NULL
        """).fetchone()[0]

        # Gender distribution
        gender_dist = conn.execute("""
            SELECT pc.poster_gender, COUNT(*) as count
            FROM post_classifications pc
            JOIN posts p ON pc.post_id = p.post_id
            WHERE p.post_id LIKE 'reddit_%'
            GROUP BY pc.poster_gender
        """).fetchall()
        stats['gender_distribution'] = {row[0]: row[1] for row in gender_dist}

        # Comments
        stats['total_comments'] = conn.execute(
            "SELECT COUNT(*) FROM comments WHERE post_id LIKE 'reddit_%'"
        ).fetchone()[0]

        stats['classified_comments'] = conn.execute("""
            SELECT COUNT(*) FROM comment_classifications cc
            JOIN comments c ON cc.comment_id = c.comment_id
            WHERE c.post_id LIKE 'reddit_%'
        """).fetchone()[0]

        # Advice direction distribution
        direction_dist = conn.execute("""
            SELECT cc.advice_direction, COUNT(*) as count
            FROM comment_classifications cc
            JOIN comments c ON cc.comment_id = c.comment_id
            WHERE c.post_id LIKE 'reddit_%'
            AND cc.is_advice = 1
            GROUP BY cc.advice_direction
        """).fetchall()
        stats['advice_direction_distribution'] = {row[0]: row[1] for row in direction_dist}

        return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Reddit Classification Pipeline")
    parser.add_argument("--step", choices=["posts", "details", "comments", "all", "stats"],
                       default="stats", help="Which step to run")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for classification")
    parser.add_argument("--dry-run", action="store_true",
                       help="Don't actually insert, just report")

    args = parser.parse_args()

    if args.step == "stats":
        stats = get_reddit_stats()
        print(json.dumps(stats, indent=2))
        return

    if args.step in ("posts", "all"):
        print("\n=== Step 1: Create post classifications ===")
        results = create_reddit_post_classifications(dry_run=args.dry_run)
        print(json.dumps(results, indent=2))

    if args.step in ("details", "all"):
        print("\n=== Step 2: Classify post details (severity/fault/category) ===")
        results = classify_reddit_post_details(batch_size=args.batch_size)
        print(json.dumps(results, indent=2))

    if args.step in ("comments", "all"):
        print("\n=== Step 3: Classify comments ===")
        results = classify_reddit_comments(batch_size=args.batch_size)
        print(json.dumps(results, indent=2))

    print("\n=== Current Stats ===")
    stats = get_reddit_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
