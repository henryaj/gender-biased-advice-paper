#!/usr/bin/env python3
"""Classify comments using Claude Code CLI instead of API."""

import sqlite3
import subprocess
import json
import time
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Unset API key to ensure we use Claude Code subscription, not API credits
if "ANTHROPIC_API_KEY" in os.environ:
    del os.environ["ANTHROPIC_API_KEY"]
    print("[INFO] Unset ANTHROPIC_API_KEY to use Claude Code subscription")

DB_PATH = Path(__file__).parent.parent / "data" / "research.db"

# Thread-local storage for database connections
thread_local = threading.local()

def get_db_connection():
    """Get thread-local database connection."""
    if not hasattr(thread_local, 'conn'):
        thread_local.conn = sqlite3.connect(DB_PATH)
        thread_local.conn.row_factory = sqlite3.Row
    return thread_local.conn

PROMPT_TEMPLATE = '''Classify this advice comment's direction toward the OP (original poster).

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

## Examples from human validation

EXAMPLE 1 - supportive_of_op:
Context: A 40-year-old man in a relationship with minimal intimacy and frequent criticism from partner
Comment: "She's resentful and snipey that you visit a terminally-ill parent 1-2 nights a week? That's actually monstrous."
Why: Validates OP, criticizes the partner's behavior

EXAMPLE 2 - supportive_of_op:
Context: A woman unsure whether to stay with her kind but emotionally distant boyfriend
Comment: "He's great except that he's emotionally cold and doesn't know you very well? I think you know the answer. Cut him loose."
Why: Validates OP's doubts, encourages her to trust her instincts

EXAMPLE 3 - critical_of_op:
Context: A 27-year-old man struggling with dating due to fixation on physical attractiveness
Comment: "One big problem is the whole notion of 'leagues'. Quit that; it's juvenile and harmful to you and anyone you might date."
Why: Directly criticizes OP's mindset as "juvenile and harmful"

EXAMPLE 4 - critical_of_op:
Context: A 4-month relationship where OP is experiencing communication challenges
Comment: "Did you do this on purpose, to make her more stressed? If not, I don't understand this at all. Why would you do the exact thing she just told you not to do?"
Why: Directly calls out OP's behavior as problematic

EXAMPLE 5 - neutral:
Context: A couple with different lifestyles struggling with compatibility
Comment: "Just go ahead & break up, this sounds like agony for both of you. It's not supposed to be this difficult."
Why: Gives practical advice without blaming either party

EXAMPLE 6 - neutral:
Context: A woman in an unclear romantic situation with a man from a dating app
Comment: "Are you sure he's not married? Meeting only on his terms, being affectionate but not going any further than that... 'he's married' was my first thought."
Why: Offers a possibility to consider, no judgment of OP

EXAMPLE 7 - mixed:
Context: A male coworker dealing with a female colleague who seems romantically interested
Comment: "I'm sorry to deprive you of a friend, but you probably shouldn't be going to dinner and movies with her and helping her with home renovation choices. She might be thinking of these as dates."
Why: Empathetic ("I'm sorry") but also points out OP's role in creating the situation

## Now classify this comment

Context: {context}

Comment: "{comment}"

Return ONLY a JSON object: {{"advice_direction": "supportive_of_op" | "critical_of_op" | "neutral" | "mixed"}}'''


def classify_comment(context: str, comment: str, model: str = "sonnet") -> tuple[str | None, str | None]:
    """Classify a single comment using claude -p.

    Returns: (direction, error_message) - one will be None
    """
    # Truncate if needed
    if len(comment) > 2000:
        comment = comment[:2000] + "..."
    if len(context) > 300:
        context = context[:300] + "..."

    prompt = PROMPT_TEMPLATE.format(context=context, comment=comment)

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
        direction = data.get("advice_direction")

        if direction in ("supportive_of_op", "critical_of_op", "neutral", "mixed"):
            return direction, None
        return None, f"Invalid direction: {direction}"

    except subprocess.TimeoutExpired:
        return None, "Timeout (120s)"
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e} - output was: {output[:100]}"
    except Exception as e:
        return None, f"Unexpected error: {type(e).__name__}: {e}"


def get_comments_to_classify(conn, limit: int = 100, validated_only: bool = False):
    """Get comments that need classification."""
    if validated_only:
        # Only get comments that have human validations
        return conn.execute('''
            SELECT c.comment_id, c.body, pc.brief_situation_summary, pc.poster_gender,
                   cc.advice_direction as old_direction
            FROM comments c
            JOIN post_classifications pc ON c.post_id = pc.post_id
            JOIN comment_classifications cc ON c.comment_id = cc.comment_id
            JOIN classification_validations cv ON c.comment_id = cv.comment_id
            WHERE cv.field_name = 'advice_direction_blind'
            AND pc.poster_gender IN ('male', 'female')
            LIMIT ?
        ''', (limit,)).fetchall()
    else:
        return conn.execute('''
            SELECT c.comment_id, c.body, pc.brief_situation_summary, pc.poster_gender,
                   cc.advice_direction as old_direction
            FROM comments c
            JOIN post_classifications pc ON c.post_id = pc.post_id
            JOIN comment_classifications cc ON c.comment_id = cc.comment_id
            WHERE cc.is_advice = 1
            AND pc.poster_gender IN ('male', 'female')
            AND pc.problem_category IS NOT NULL
            AND cc.classification_model NOT LIKE 'claude-code-%'
            LIMIT ?
        ''', (limit,)).fetchall()


def process_comment(comment_data, model):
    """Process a single comment (for parallel execution)."""
    comment_id, body, context, gender, old_direction = comment_data
    context = context or "No context"

    new_direction, error = classify_comment(context, body, model=model)

    if error:
        return {"status": "error", "comment_id": comment_id, "error": error}

    # Update database using thread-local connection
    conn = get_db_connection()
    conn.execute('''
        UPDATE comment_classifications
        SET advice_direction = ?, classification_model = ?
        WHERE comment_id = ?
    ''', (new_direction, f'claude-code-{model}', comment_id))
    conn.commit()

    return {
        "status": "success",
        "comment_id": comment_id,
        "old": old_direction,
        "new": new_direction,
        "gender": gender,
        "changed": old_direction != new_direction
    }


def main():
    import datetime

    # Parse args
    validated_only = "--validated" in sys.argv
    exit_on_error = "--exit-on-error" in sys.argv
    model = "sonnet"  # Default to sonnet (cheaper)
    workers = 10  # Default parallel workers

    # Parse --model and --workers
    for i, arg in enumerate(sys.argv):
        if arg == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
        if arg == "--workers" and i + 1 < len(sys.argv):
            workers = int(sys.argv[i + 1])

    args = [a for a in sys.argv[1:] if not a.startswith("--") and sys.argv[sys.argv.index(a)-1] not in ("--model", "--workers")]
    batch_size = int(args[0]) if args else 100

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    comments = get_comments_to_classify(conn, batch_size, validated_only=validated_only)
    conn.close()

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting classification")
    print(f"  Model: {model}")
    print(f"  Workers: {workers}")
    print(f"  Comments: {len(comments)}")
    print(f"  Validated only: {validated_only}")
    print(f"  Exit on error: {exit_on_error}")
    print()

    # Prepare data for parallel processing
    comment_data = [
        (c["comment_id"], c["body"], c["brief_situation_summary"], c["poster_gender"], c["old_direction"])
        for c in comments
    ]

    results = {"unchanged": 0, "changed": 0, "errors": 0}
    changes = []
    errors = []
    completed = 0
    start_time = datetime.datetime.now()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_comment, cd, model): cd for cd in comment_data}

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            if result["status"] == "error":
                results["errors"] += 1
                errors.append(result)
                print(f"[ERROR] {result['comment_id']}: {result.get('error', 'Unknown error')}")

                if exit_on_error:
                    print("\n--exit-on-error set, stopping...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

            elif result["changed"]:
                results["changed"] += 1
                changes.append(result)
                print(f"[CHANGE] {result['old']:20} -> {result['new']:20} ({result['gender']})")
            else:
                results["unchanged"] += 1

            # Progress every 25 or every 30 seconds
            if completed % 25 == 0:
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(comments) - completed) / rate if rate > 0 else 0
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Progress: {completed}/{len(comments)} "
                      f"({100*completed/len(comments):.1f}%) | "
                      f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m | "
                      f"Changed: {results['changed']} | Errors: {results['errors']}")

    # Summarize
    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    print(f"\n{'='*60}")
    print(f"COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  Processed: {completed}")
    print(f"  Unchanged: {results['unchanged']}")
    print(f"  Changed:   {results['changed']}")
    print(f"  Errors:    {results['errors']}")

    if changes:
        change_summary = {}
        for c in changes:
            key = f"{c['old']} -> {c['new']}"
            if key not in change_summary:
                change_summary[key] = {"total": 0, "male": 0, "female": 0}
            change_summary[key]["total"] += 1
            if c["gender"] in ("male", "female"):
                change_summary[key][c["gender"]] += 1

        print("\nChange breakdown:")
        for key, counts in sorted(change_summary.items(), key=lambda x: -x[1]["total"]):
            print(f"  {key}: {counts['total']} (M:{counts['male']}, F:{counts['female']})")

    if errors:
        print(f"\nFirst 5 errors:")
        for e in errors[:5]:
            print(f"  {e['comment_id']}: {e.get('error', 'Unknown')[:80]}")

    # Exit with error code if there were errors
    if results["errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
