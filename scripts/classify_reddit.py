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
        print(f"  ✗ {post_id}: {error}", flush=True)
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

    print(f"  ✓ {post_id}: {result.get('situation_severity')}/{result.get('op_fault')}/{result.get('problem_category')}", flush=True)

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


# Reddit-specific prompt - tuned for Reddit's more direct/blunt style
REDDIT_COMMENT_PROMPT = '''Classify this Reddit advice comment's direction toward the OP (original poster).

## Definitions

- "supportive_of_op": EXPLICITLY validates OP emotionally, says things like "you're right", "you deserve better", "that's unfair", or STRONGLY takes OP's side with emotional language
- "critical_of_op": Directly criticizes OP's behavior, decisions, mindset, or perspective
- "neutral": Practical advice, factual information, or blunt suggestions WITHOUT emotional validation. Includes matter-of-fact statements like "leave him" or "change your passwords" that don't explicitly take sides emotionally
- "mixed": Contains BOTH supportive AND critical elements toward OP

KEY DISTINCTION: On Reddit, many comments give blunt advice ("dump him", "run", "leave") without emotional validation. These are NEUTRAL - they're practical suggestions, not emotional support. "Supportive" requires EXPLICIT emotional validation or strong advocacy language, not just implied agreement.

Examples of NEUTRAL (not supportive):
- "Change your passwords and secure your accounts" (practical info)
- "Get rid of him" (blunt suggestion without emotional content)
- "They're gonna bang eventually" (observation/prediction)

Examples of SUPPORTIVE:
- "You deserve so much better than this" (emotional validation)
- "NTA, your boyfriend is being completely unreasonable" (explicit judgment + validation)
- "I'm so sorry you're dealing with this, you did nothing wrong" (emotional support)

## Context
{context}

## Comment
"{comment}"

Return ONLY: {{"is_advice": true/false, "advice_direction": "supportive_of_op" | "critical_of_op" | "neutral" | "mixed" | null}}'''


def classify_single_comment(comment_data: tuple, model: str, use_reddit_prompt: bool = False) -> dict:
    """Classify a single comment (for parallel execution)."""
    comment_id, body, post_id, context, gender = comment_data

    # Truncate if needed
    if len(body) > 2000:
        body = body[:2000] + "..."
    if len(context) > 300:
        context = context[:300] + "..."

    # Use Reddit-specific prompt if specified
    base_prompt = REDDIT_COMMENT_PROMPT if use_reddit_prompt else COMMENT_PROMPT
    prompt = base_prompt.format(context=context, comment=body)

    result, error = claude_classify(prompt, model=model)

    if error:
        print(f"  ✗ {comment_id}: {error[:50]}", flush=True)
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
    print(f"\n{'='*60}", flush=True)
    print(f"CLASSIFY REDDIT COMMENTS", flush=True)
    print(f"Model: {model} | Workers: {workers} | Batch: {batch_size}", flush=True)
    print(f"{'='*60}\n", flush=True)

    with db.get_connection() as conn:
        # Randomly sample comments across all posts to avoid clustering
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
            ORDER BY RANDOM()
            LIMIT ?
        """, (batch_size,)).fetchall()

    print(f"Found {len(comments)} Reddit comments to classify\n", flush=True)

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
        # Use Reddit-specific prompt for better accuracy
        futures = {executor.submit(classify_single_comment, cd, model, use_reddit_prompt=True): cd for cd in comment_data}

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

            # Progress every 10
            if classified % 10 == 0 and classified > 0:
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                rate = classified / elapsed if elapsed > 0 else 0
                eta = (len(comments) - classified) / rate if rate > 0 else 0
                print(f"[{classified}/{len(comments)}] {100*classified/len(comments):.1f}% | {rate:.1f}/s | ETA {eta/60:.1f}m | {errors} errors | advice: {advice_count}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"✓ COMPLETE: {classified} comments classified", flush=True)
    print(f"  Advice: {advice_count} | Errors: {errors}", flush=True)
    print(f"  Directions: {direction_counts}", flush=True)
    print(f"{'='*60}\n", flush=True)

    return {
        "classified": classified,
        "advice_count": advice_count,
        "errors": errors,
        "direction_counts": direction_counts
    }


def reclassify_post_details(
    source: str = "mf_%",
    batch_size: int = 100,
    model: str = "sonnet",
    workers: int = 5
) -> dict:
    """
    Reclassify post details (severity/fault/category) for existing posts.

    Args:
        source: Post ID pattern ('mf_%' for Metafilter, 'reddit_%' for Reddit)
    """
    print(f"\n{'='*60}", flush=True)
    print(f"RECLASSIFY POST DETAILS", flush=True)
    print(f"Source: {source} | Model: {model} | Workers: {workers} | Batch: {batch_size}", flush=True)
    print(f"{'='*60}\n", flush=True)

    with db.get_connection() as conn:
        posts = conn.execute("""
            SELECT p.post_id, p.title, p.body, pc.poster_gender
            FROM posts p
            JOIN post_classifications pc ON p.post_id = pc.post_id
            WHERE p.post_id LIKE ?
            AND pc.poster_gender IN ('male', 'female')
            LIMIT ?
        """, (source, batch_size)).fetchall()

    print(f"Found {len(posts)} posts to reclassify\n", flush=True)

    if not posts:
        return {"classified": 0, "errors": 0}

    post_data = [(p['post_id'], p['title'], p['body'], p['poster_gender']) for p in posts]

    classified = 0
    errors = 0

    import datetime
    start_time = datetime.datetime.now()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(classify_single_post, pd, model): pd for pd in post_data}

        for future in as_completed(futures):
            result = future.result()

            if result["status"] == "error":
                errors += 1
            else:
                classified += 1

            # Progress every 5
            if (classified + errors) % 5 == 0:
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                rate = (classified + errors) / elapsed if elapsed > 0 else 0
                remaining = len(posts) - classified - errors
                eta = remaining / rate if rate > 0 else 0
                print(f"[{classified + errors}/{len(posts)}] {100*(classified + errors)/len(posts):.1f}% | {rate:.1f}/s | ETA {eta/60:.1f}m | {errors} errors", flush=True)

    print(f"\n✓ Reclassified {classified} posts, {errors} errors", flush=True)
    return {"classified": classified, "errors": errors}


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


# ============================================================================
# Gender Inference for posts without explicit markers
# ============================================================================

GENDER_INFERENCE_PROMPT = '''Analyze this relationship advice post and determine the poster's gender.

<post>
Title: {title}

Body:
{body}
</post>

Look for clues:
- First-person references to self (e.g., "I (25M)" or gendered language)
- References to partner's gender which may imply OP's gender (e.g., "my girlfriend" often implies male OP)
- Contextual clues about the relationship dynamic
- Gendered terms used to describe self

IMPORTANT: Only return male/female if you are reasonably confident. If unclear, return "unknown".

Return ONLY a JSON object:
{{"poster_gender": "male" | "female" | "unknown", "confidence": 0.5-0.95, "reasoning": "brief explanation"}}'''


def infer_single_gender(post_data: tuple, model: str) -> dict:
    """Infer gender for a single post (for parallel execution)."""
    post_id, title, body = post_data

    prompt = GENDER_INFERENCE_PROMPT.format(
        title=title,
        body=body[:6000] if body else ""
    )

    result, error = claude_classify(prompt, model=model)

    if error:
        print(f"  ✗ {post_id}: {error[:50]}", flush=True)
        return {"status": "error", "post_id": post_id, "error": error}

    gender = result.get('poster_gender', 'unknown')
    confidence = result.get('confidence', 0.5)

    # Validate
    if gender not in ('male', 'female', 'unknown'):
        gender = 'unknown'
    confidence = min(0.85, max(0.5, confidence))  # Cap at 0.85 for inferred gender

    # Update database
    conn = get_thread_db()
    conn.execute("""
        UPDATE post_classifications
        SET poster_gender = ?,
            gender_confidence = ?,
            classification_model = ?
        WHERE post_id = ?
        AND (poster_gender = 'unknown' OR poster_gender IS NULL)
    """, (gender, confidence, f'claude-code-{model}-inferred', post_id))
    conn.commit()

    print(f"  ✓ {post_id}: {gender} ({confidence:.2f}) - {result.get('reasoning', '')[:50]}", flush=True)

    return {
        "status": "success",
        "post_id": post_id,
        "gender": gender,
        "confidence": confidence
    }


def infer_reddit_genders(
    batch_size: int = 100,
    model: str = "sonnet",
    workers: int = 5
) -> dict:
    """
    Infer gender for Reddit posts that don't have explicit markers.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"INFER GENDER FOR POSTS WITHOUT MARKERS", flush=True)
    print(f"Model: {model} | Workers: {workers} | Batch: {batch_size}", flush=True)
    print(f"{'='*60}\n", flush=True)

    with db.get_connection() as conn:
        # Find posts without gender or with unknown gender
        posts = conn.execute("""
            SELECT p.post_id, p.title, p.body
            FROM posts p
            JOIN post_classifications pc ON p.post_id = pc.post_id
            WHERE p.post_id LIKE 'reddit_%'
            AND (pc.poster_gender = 'unknown' OR pc.poster_gender IS NULL)
            LIMIT ?
        """, (batch_size,)).fetchall()

    print(f"Found {len(posts)} posts needing gender inference\n", flush=True)

    if not posts:
        return {"inferred": 0, "errors": 0}

    post_data = [(p['post_id'], p['title'], p['body']) for p in posts]

    inferred = 0
    errors = 0
    gender_counts = {"male": 0, "female": 0, "unknown": 0}

    import datetime
    start_time = datetime.datetime.now()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(infer_single_gender, pd, model): pd for pd in post_data}

        for future in as_completed(futures):
            result = future.result()

            if result["status"] == "error":
                errors += 1
            else:
                inferred += 1
                gender_counts[result["gender"]] += 1

            # Progress every 10
            if (inferred + errors) % 10 == 0 and (inferred + errors) > 0:
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                rate = (inferred + errors) / elapsed if elapsed > 0 else 0
                remaining = len(posts) - inferred - errors
                eta = remaining / rate if rate > 0 else 0
                print(f"[{inferred + errors}/{len(posts)}] {100*(inferred + errors)/len(posts):.1f}% | {rate:.1f}/s | ETA {eta/60:.1f}m", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"✓ COMPLETE: {inferred} genders inferred", flush=True)
    print(f"  Male: {gender_counts['male']} | Female: {gender_counts['female']} | Unknown: {gender_counts['unknown']}", flush=True)
    print(f"  Errors: {errors}", flush=True)
    print(f"{'='*60}\n", flush=True)

    return {
        "inferred": inferred,
        "errors": errors,
        "gender_counts": gender_counts
    }


def reclassify_validated_reddit_comments(model: str = "sonnet", workers: int = 10) -> dict:
    """
    Reclassify Reddit comments that have been validated using the Reddit-specific prompt.
    This tests the new prompt against human judgments.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"RECLASSIFY VALIDATED REDDIT COMMENTS", flush=True)
    print(f"Model: {model} | Workers: {workers}", flush=True)
    print(f"Using Reddit-specific prompt", flush=True)
    print(f"{'='*60}\n", flush=True)

    with db.get_connection() as conn:
        # Get validated Reddit comments
        comments = conn.execute("""
            SELECT DISTINCT c.comment_id, c.body, c.post_id,
                   COALESCE(pc.brief_situation_summary, p.title) as context,
                   pc.poster_gender
            FROM classification_validations cv
            JOIN comments c ON cv.comment_id = c.comment_id
            JOIN posts p ON c.post_id = p.post_id
            JOIN post_classifications pc ON p.post_id = pc.post_id
            WHERE c.post_id LIKE 'reddit_%'
        """).fetchall()

    print(f"Found {len(comments)} validated Reddit comments to reclassify\n", flush=True)

    if not comments:
        return {"reclassified": 0, "errors": 0}

    comment_data = [
        (c['comment_id'], c['body'], c['post_id'], c['context'], c['poster_gender'])
        for c in comments
    ]

    reclassified = 0
    errors = 0
    direction_counts = {
        "supportive_of_op": 0,
        "critical_of_op": 0,
        "neutral": 0,
        "mixed": 0
    }

    import datetime
    from functools import partial
    start_time = datetime.datetime.now()

    # Use Reddit-specific prompt
    classify_fn = partial(classify_single_comment, model=model, use_reddit_prompt=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(classify_fn, cd): cd for cd in comment_data}

        for future in as_completed(futures):
            result = future.result()

            if result["status"] == "error":
                errors += 1
            else:
                reclassified += 1
                if result.get("is_advice") and result.get("direction"):
                    direction_counts[result["direction"]] += 1

            # Progress
            total = reclassified + errors
            if total % 10 == 0 and total > 0:
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                rate = total / elapsed if elapsed > 0 else 0
                print(f"[{total}/{len(comments)}] {100*total/len(comments):.1f}% | {rate:.1f}/s", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"✓ COMPLETE: {reclassified} comments reclassified", flush=True)
    print(f"  Errors: {errors}", flush=True)
    print(f"  Directions: {direction_counts}", flush=True)
    print(f"{'='*60}\n", flush=True)

    return {
        "reclassified": reclassified,
        "errors": errors,
        "direction_counts": direction_counts
    }


def reclassify_all_reddit_comments(batch_size: int = 1000, model: str = "sonnet", workers: int = 10) -> dict:
    """
    Reclassify ALL Reddit comments using the Reddit-specific prompt.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"RECLASSIFY ALL REDDIT COMMENTS", flush=True)
    print(f"Model: {model} | Workers: {workers} | Batch: {batch_size}", flush=True)
    print(f"Using Reddit-specific prompt", flush=True)
    print(f"{'='*60}\n", flush=True)

    with db.get_connection() as conn:
        # Get ALL classified Reddit comments
        comments = conn.execute("""
            SELECT c.comment_id, c.body, c.post_id,
                   COALESCE(pc.brief_situation_summary, p.title) as context,
                   pc.poster_gender
            FROM comments c
            JOIN posts p ON c.post_id = p.post_id
            JOIN post_classifications pc ON p.post_id = pc.post_id
            JOIN comment_classifications cc ON c.comment_id = cc.comment_id
            WHERE c.post_id LIKE 'reddit_%'
            AND pc.poster_gender IN ('male', 'female')
            LIMIT ?
        """, (batch_size,)).fetchall()

    print(f"Found {len(comments)} Reddit comments to reclassify\n", flush=True)

    if not comments:
        return {"reclassified": 0, "errors": 0}

    comment_data = [
        (c['comment_id'], c['body'], c['post_id'], c['context'], c['poster_gender'])
        for c in comments
    ]

    reclassified = 0
    errors = 0
    direction_counts = {
        "supportive_of_op": 0,
        "critical_of_op": 0,
        "neutral": 0,
        "mixed": 0
    }

    import datetime
    from functools import partial
    start_time = datetime.datetime.now()

    classify_fn = partial(classify_single_comment, model=model, use_reddit_prompt=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(classify_fn, cd): cd for cd in comment_data}

        for future in as_completed(futures):
            result = future.result()

            if result["status"] == "error":
                errors += 1
            else:
                reclassified += 1
                if result.get("is_advice") and result.get("direction"):
                    direction_counts[result["direction"]] += 1

            total = reclassified + errors
            if total % 50 == 0 and total > 0:
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                rate = total / elapsed if elapsed > 0 else 0
                eta = (len(comments) - total) / rate if rate > 0 else 0
                print(f"[{total}/{len(comments)}] {100*total/len(comments):.1f}% | {rate:.1f}/s | ETA {eta/60:.1f}m | {errors} errors", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"✓ COMPLETE: {reclassified} comments reclassified", flush=True)
    print(f"  Errors: {errors}", flush=True)
    print(f"  Directions: {direction_counts}", flush=True)
    print(f"{'='*60}\n", flush=True)

    return {
        "reclassified": reclassified,
        "errors": errors,
        "direction_counts": direction_counts
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Classification Pipeline (uses claude -p CLI)")
    parser.add_argument("--step", choices=["posts", "details", "comments", "all", "stats", "reclassify-mf", "reclassify-reddit", "infer-gender", "scrape", "test-reddit-prompt", "reclassify-reddit-comments"],
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

    if args.step == "reclassify-mf":
        print("\n=== Reclassifying Metafilter posts ===")
        results = reclassify_post_details(
            source="mf_%",
            batch_size=args.batch_size,
            model=args.model,
            workers=args.workers
        )
        print(json.dumps(results, indent=2))

    if args.step == "reclassify-reddit":
        print("\n=== Reclassifying Reddit posts ===")
        results = reclassify_post_details(
            source="reddit_%",
            batch_size=args.batch_size,
            model=args.model,
            workers=args.workers
        )
        print(json.dumps(results, indent=2))

    if args.step == "infer-gender":
        print("\n=== Inferring gender for posts without markers ===")
        results = infer_reddit_genders(
            batch_size=args.batch_size,
            model=args.model,
            workers=args.workers
        )
        print(json.dumps(results, indent=2))

    if args.step == "scrape":
        print("\n=== Scraping Reddit posts (without gender marker requirement) ===")
        from src.scrapers.reddit_scraper import RedditJSONScraper
        scraper = RedditJSONScraper()
        scraper.checkpoint['after'] = None  # Reset cursor
        scraper._save_checkpoint()
        results = scraper.scrape(
            subreddit='relationship_advice',
            max_posts=args.batch_size,
            min_comments=5,
            time_filter='year',
            sort='top',
            require_gender_marker=False  # Include posts without markers
        )
        print(json.dumps(results, indent=2))

        # Also create post classifications for new posts
        print("\n=== Creating post classifications for new posts ===")
        create_reddit_post_classifications(dry_run=args.dry_run)

    if args.step == "test-reddit-prompt":
        print("\n=== Testing Reddit-specific prompt on validated comments ===")
        results = reclassify_validated_reddit_comments(
            model=args.model,
            workers=args.workers
        )
        print(json.dumps(results, indent=2))

    if args.step == "reclassify-reddit-comments":
        print("\n=== Reclassifying ALL Reddit comments with Reddit-specific prompt ===")
        results = reclassify_all_reddit_comments(
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
