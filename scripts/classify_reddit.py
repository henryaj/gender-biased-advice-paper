#!/usr/bin/env python3
"""
Reddit Classification Pipeline

For Reddit posts from r/relationship_advice:
1. Gender is already extracted during scraping (stored in flair field)
2. is_relationship_advice is always True (all from relationship subreddit)
3. Still need LLM for: situation_severity, op_fault, problem_category
4. Comments need full LLM classification (same as Metafilter)

Uses 'claude -p' CLI to avoid API key requirements (uses Claude Code subscription).
"""

import sys
import os
import re
import json
import subprocess
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Unset API key to ensure we use Claude Code subscription, not API credits
if "ANTHROPIC_API_KEY" in os.environ:
    del os.environ["ANTHROPIC_API_KEY"]

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import db
from src.utils.logging import get_logger

logger = get_logger("classify_reddit")

# Thread-local storage for database connections
thread_local = threading.local()

def get_thread_db():
    """Get thread-local database connection."""
    if not hasattr(thread_local, 'conn'):
        import sqlite3
        db_path = Path(__file__).parent.parent / "data" / "research.db"
        thread_local.conn = sqlite3.connect(db_path)
        thread_local.conn.row_factory = sqlite3.Row
    return thread_local.conn


def claude_classify(prompt: str, model: str = "sonnet") -> tuple[dict | None, str | None]:
    """
    Classify using claude -p CLI.

    Returns: (result_dict, error_message) - one will be None
    """
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", model],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            return None, f"CLI error (code {result.returncode}): {result.stderr[:200]}"

        output = result.stdout.strip()

        if not output:
            return None, "Empty response from CLI"

        # Extract JSON from response (may be wrapped in ```json```)
        if "```" in output:
            lines = output.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            output = "\n".join(json_lines)

        data = json.loads(output)
        return data, None

    except subprocess.TimeoutExpired:
        return None, "Timeout (120s)"
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
    except Exception as e:
        return None, f"Unexpected error: {type(e).__name__}: {e}"


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


POST_DETAILS_PROMPT = '''Analyze this relationship advice post and extract:

<post>
Title: {title}

Body:
{body}
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

{{"situation_severity": "...", "op_fault": "...", "problem_category": "...", "situation_summary": "..."}}'''


def classify_single_post(post_data: tuple, model: str) -> dict:
    """Classify a single post (for parallel execution)."""
    post_id, title, body, gender = post_data

    prompt = POST_DETAILS_PROMPT.format(
        title=title,
        body=body[:6000] if body else ""
    )

    result, error = claude_classify(prompt, model=model)

    if error:
        return {"status": "error", "post_id": post_id, "error": error}

    # Update database using thread-local connection
    conn = get_thread_db()
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
        f'claude-code-{model}',
        post_id
    ))
    conn.commit()

    return {
        "status": "success",
        "post_id": post_id,
        "severity": result.get('situation_severity'),
        "fault": result.get('op_fault'),
        "category": result.get('problem_category')
    }


def classify_reddit_post_details(
    batch_size: int = 100,
    model: str = "haiku",
    workers: int = 5
) -> dict:
    """
    Classify situation_severity, op_fault, and problem_category for Reddit posts.

    Uses claude -p CLI (no API key needed).
    """
    logger.info(f"Classifying Reddit post details with {model} ({workers} workers)...")

    with db.get_connection() as conn:
        posts = conn.execute("""
            SELECT p.post_id, p.title, p.body, pc.poster_gender
            FROM posts p
            JOIN post_classifications pc ON p.post_id = pc.post_id
            WHERE p.post_id LIKE 'reddit_%'
            AND pc.situation_severity IS NULL
            LIMIT ?
        """, (batch_size,)).fetchall()

    logger.info(f"Found {len(posts)} Reddit posts needing detail classification")

    if not posts:
        return {"classified": 0, "errors": 0}

    post_data = [(p['post_id'], p['title'], p['body'], p['poster_gender']) for p in posts]

    classified = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(classify_single_post, pd, model): pd for pd in post_data}

        for future in as_completed(futures):
            result = future.result()

            if result["status"] == "error":
                errors += 1
                logger.error(f"Error: {result['post_id']}: {result['error']}")
            else:
                classified += 1
                logger.info(f"Classified {result['post_id']}: {result['severity']}/{result['fault']}/{result['category']}")

    logger.info(f"Classified {classified} posts, {errors} errors")
    return {"classified": classified, "errors": errors}


COMMENT_PROMPT = '''Classify this advice comment's direction toward the OP (original poster).

## Definitions

- "supportive_of_op": Validates OP's perspective, sides with them, supports their instincts, criticizes the OTHER party
- "critical_of_op": Directly criticizes OP's behavior, decisions, mindset, or perspective. Must explicitly call out something OP did wrong.
- "neutral": Gives advice without taking sides or judging either party
- "mixed": Contains BOTH supportive AND critical elements toward OP

IMPORTANT: Be conservative with "critical_of_op". Most advice that seems negative is actually:
- Supportive (criticizing OP's partner/situation, not OP)
- Neutral (giving practical advice without judgment)
- Mixed (acknowledging complexity)

Only use "critical_of_op" when the comment DIRECTLY criticizes OP's own actions, decisions, or character.

## Context
{context}

## Comment
"{comment}"

Return ONLY: {{"is_advice": true/false, "advice_direction": "supportive_of_op" | "critical_of_op" | "neutral" | "mixed" | null}}'''


def classify_single_comment(comment_data: tuple, model: str) -> dict:
    """Classify a single comment (for parallel execution)."""
    comment_id, body, post_id, context, gender = comment_data

    # Truncate if needed
    if len(body) > 2000:
        body = body[:2000] + "..."
    if len(context) > 300:
        context = context[:300] + "..."

    prompt = COMMENT_PROMPT.format(context=context, comment=body)

    result, error = claude_classify(prompt, model=model)

    if error:
        return {"status": "error", "comment_id": comment_id, "error": error}

    is_advice = result.get("is_advice", False)
    direction = result.get("advice_direction")

    # Validate direction
    if direction not in ("supportive_of_op", "critical_of_op", "neutral", "mixed", None):
        direction = None

    # Insert using thread-local connection
    conn = get_thread_db()
    conn.execute("""
        INSERT OR REPLACE INTO comment_classifications
        (comment_id, is_advice, advice_direction, tone_labels, classification_model)
        VALUES (?, ?, ?, ?, ?)
    """, (comment_id, is_advice, direction, "[]", f'claude-code-{model}'))
    conn.commit()

    return {
        "status": "success",
        "comment_id": comment_id,
        "is_advice": is_advice,
        "direction": direction,
        "gender": gender
    }


def classify_reddit_comments(
    batch_size: int = 1000,
    model: str = "sonnet",
    workers: int = 10
) -> dict:
    """
    Classify comments on Reddit posts using claude -p CLI.
    """
    logger.info(f"Classifying Reddit comments with {model} ({workers} workers)...")

    with db.get_connection() as conn:
        comments = conn.execute("""
            SELECT c.comment_id, c.body, c.post_id,
                   COALESCE(pc.brief_situation_summary, p.title) as context,
                   pc.poster_gender
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

    if not comments:
        return {"classified": 0, "advice_count": 0, "errors": 0, "direction_counts": {}}

    comment_data = [
        (c['comment_id'], c['body'], c['post_id'], c['context'], c['poster_gender'])
        for c in comments
    ]

    classified = 0
    advice_count = 0
    errors = 0
    direction_counts = {
        "supportive_of_op": 0,
        "critical_of_op": 0,
        "neutral": 0,
        "mixed": 0
    }

    import datetime
    start_time = datetime.datetime.now()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(classify_single_comment, cd, model): cd for cd in comment_data}

        for future in as_completed(futures):
            result = future.result()

            if result["status"] == "error":
                errors += 1
                if errors <= 5:
                    logger.error(f"Error: {result['comment_id']}: {result['error']}")
            else:
                classified += 1
                if result["is_advice"]:
                    advice_count += 1
                    if result["direction"]:
                        direction_counts[result["direction"]] += 1

            # Progress every 50
            if classified % 50 == 0 and classified > 0:
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                rate = classified / elapsed if elapsed > 0 else 0
                eta = (len(comments) - classified) / rate if rate > 0 else 0
                logger.info(f"Progress: {classified}/{len(comments)} ({100*classified/len(comments):.1f}%) | "
                           f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m | Errors: {errors}")

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

    parser = argparse.ArgumentParser(description="Reddit Classification Pipeline (uses claude -p CLI)")
    parser.add_argument("--step", choices=["posts", "details", "comments", "all", "stats"],
                       default="stats", help="Which step to run")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for classification")
    parser.add_argument("--model", default="sonnet",
                       help="Model to use: sonnet, haiku, opus (default: sonnet)")
    parser.add_argument("--workers", type=int, default=10,
                       help="Number of parallel workers (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Don't actually insert, just report")

    args = parser.parse_args()

    print(f"Using model: {args.model}, workers: {args.workers}")

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
        results = classify_reddit_post_details(
            batch_size=args.batch_size,
            model=args.model,
            workers=args.workers
        )
        print(json.dumps(results, indent=2))

    if args.step in ("comments", "all"):
        print("\n=== Step 3: Classify comments ===")
        results = classify_reddit_comments(
            batch_size=args.batch_size,
            model=args.model,
            workers=args.workers
        )
        print(json.dumps(results, indent=2))

    print("\n=== Current Stats ===")
    stats = get_reddit_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
