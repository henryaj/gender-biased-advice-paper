"""Comment classifier using Claude API to analyze advice tone and direction."""

import os
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import anthropic

from ..utils.database import db
from ..utils.logging import get_logger, api_logger

logger = get_logger("comment_classifier")

# Default model for classification
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Classification prompt template with few-shot examples from human validation
COMMENT_CLASSIFICATION_PROMPT = """Analyze this comment responding to an advice forum post. Classify the advice direction.

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

<example direction="supportive_of_op">
Post context: A 40-year-old man in a relationship with minimal intimacy and frequent criticism from partner
Comment: "She's resentful and snipey that you visit a terminally-ill parent 1-2 nights a week? That's actually monstrous."
Why supportive: Validates OP, criticizes the partner's behavior
</example>

<example direction="supportive_of_op">
Post context: A woman unsure whether to stay with her kind but emotionally distant boyfriend
Comment: "He's great except that he's emotionally cold and doesn't know you very well? I think you know the answer. Cut him loose."
Why supportive: Validates OP's doubts, encourages her to trust her instincts
</example>

<example direction="critical_of_op">
Post context: A 27-year-old man struggling with dating due to fixation on physical attractiveness
Comment: "One big problem is the whole notion of 'leagues'. Quit that; it's juvenile and harmful to you and anyone you might date."
Why critical: Directly criticizes OP's mindset as "juvenile and harmful"
</example>

<example direction="critical_of_op">
Post context: A 4-month relationship where OP is experiencing communication challenges
Comment: "Did you do this on purpose, to make her more stressed? If not, I don't understand this at all. Why would you do the exact thing she just told you not to do?"
Why critical: Directly calls out OP's behavior as problematic
</example>

<example direction="neutral">
Post context: A couple with different lifestyles struggling with compatibility
Comment: "Just go ahead & break up, this sounds like agony for both of you. It's not supposed to be this difficult."
Why neutral: Gives practical advice without blaming either party
</example>

<example direction="neutral">
Post context: A woman in an unclear romantic situation with a man from a dating app
Comment: "Are you sure he's not married? Meeting only on his terms, being affectionate but not going any further than that... 'he's married' was my first thought."
Why neutral: Offers a possibility to consider, no judgment of OP
</example>

<example direction="mixed">
Post context: A male coworker dealing with a female colleague who seems romantically interested
Comment: "I'm sorry to deprive you of a friend, but you probably shouldn't be going to dinner and movies with her and helping her with home renovation choices. She might be thinking of these as dates."
Why mixed: Empathetic ("I'm sorry") but also points out OP's role in creating the situation
</example>

## Now classify this comment

<original_post_context>
{post_context}
</original_post_context>

<comment_to_analyze>
{comment}
</comment_to_analyze>

Return ONLY a JSON object:

{{
  "is_advice": true/false,
  "advice_direction": "supportive_of_op" | "critical_of_op" | "neutral" | "mixed" | null,
  "tone_labels": []
}}"""


@dataclass
class CommentClassification:
    """Classification result for a comment."""
    is_advice: bool
    advice_direction: Optional[str]
    tone_labels: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommentClassification":
        return cls(
            is_advice=data.get("is_advice", False),
            advice_direction=data.get("advice_direction"),
            tone_labels=data.get("tone_labels", [])
        )


class CommentClassifier:
    """Classifier for advice comments using Claude API."""

    VALID_DIRECTIONS = {"supportive_of_op", "critical_of_op", "neutral", "mixed", None}
    VALID_TONES = {
        "harsh", "gentle", "empathetic", "judgmental", "constructive",
        "dismissive", "understanding", "condescending", "encouraging",
        "blaming", "supportive", "hostile"
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the classifier.

        Args:
            model: Claude model to use
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            max_retries: Maximum number of retries on API errors
            retry_delay: Base delay between retries (exponential backoff)
        """
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Comment classifier initialized with model {model}")

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from model response, handling common issues."""
        text = text.strip()

        # If wrapped in markdown code blocks, extract
        if text.startswith("```"):
            lines = text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            text = "\n".join(json_lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise

    def _validate_classification(self, result: CommentClassification) -> CommentClassification:
        """Validate and clean up classification result."""
        # Validate advice_direction
        if result.advice_direction not in self.VALID_DIRECTIONS:
            result.advice_direction = None

        # Filter to valid tone labels
        result.tone_labels = [t for t in result.tone_labels if t in self.VALID_TONES]

        # If not advice, clear direction
        if not result.is_advice:
            result.advice_direction = None

        return result

    def classify_comment(
        self,
        comment_body: str,
        post_context: str,
        comment_id: Optional[str] = None
    ) -> CommentClassification:
        """
        Classify a single comment.

        Args:
            comment_body: The comment text
            post_context: Brief context about the original post
            comment_id: Optional comment ID for logging

        Returns:
            CommentClassification result
        """
        # Truncate if needed
        max_comment_length = 4000
        if len(comment_body) > max_comment_length:
            comment_body = comment_body[:max_comment_length] + "\n\n[truncated]"

        max_context_length = 500
        if len(post_context) > max_context_length:
            post_context = post_context[:max_context_length] + "..."

        prompt = COMMENT_CLASSIFICATION_PROMPT.format(
            post_context=post_context,
            comment=comment_body
        )

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}]
                )

                response_text = response.content[0].text

                # Log the API call
                api_logger.log_call(
                    endpoint="messages.create",
                    model=self.model,
                    prompt=prompt,
                    response=response_text,
                    metadata={"comment_id": comment_id, "classification_type": "comment"}
                )

                # Parse and validate response
                result_dict = self._parse_json_response(response_text)
                result = CommentClassification.from_dict(result_dict)
                return self._validate_classification(result)

            except anthropic.RateLimitError:
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"Rate limited, waiting {delay}s before retry")
                time.sleep(delay)

            except anthropic.APIError as e:
                logger.error(f"API error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    raise

            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                return CommentClassification(
                    is_advice=False,
                    advice_direction=None,
                    tone_labels=[]
                )

        return CommentClassification(
            is_advice=False,
            advice_direction=None,
            tone_labels=[]
        )

    def classify_unclassified_comments(
        self,
        batch_size: int = 1000,
        delay_between_requests: float = 0.1,
        checkpoint_every: int = 100,
        relationship_posts_only: bool = True
    ) -> Dict[str, int]:
        """
        Classify all comments that haven't been classified yet.

        Args:
            batch_size: Number of comments to process
            delay_between_requests: Delay between API calls
            checkpoint_every: Log progress every N comments
            relationship_posts_only: Only classify comments on relationship advice posts

        Returns:
            Dictionary with counts
        """
        logger.info("Starting batch comment classification")

        comments = db.get_unclassified_comments(
            relationship_posts_only=relationship_posts_only,
            limit=batch_size
        )
        logger.info(f"Found {len(comments)} unclassified comments")

        classified = 0
        advice_comments = 0
        errors = 0

        direction_counts = {
            "supportive_of_op": 0,
            "critical_of_op": 0,
            "neutral": 0,
            "mixed": 0
        }

        for comment in comments:
            try:
                # Build post context
                post_context = comment.get("brief_situation_summary", "")
                if not post_context:
                    post_context = f"Title: {comment.get('post_title', 'Unknown')}"

                result = self.classify_comment(
                    comment_body=comment["body"],
                    post_context=post_context,
                    comment_id=comment["comment_id"]
                )

                # Store classification
                db.insert_comment_classification(
                    comment_id=comment["comment_id"],
                    is_advice=result.is_advice,
                    advice_direction=result.advice_direction,
                    tone_labels=result.tone_labels,
                    classification_model=self.model
                )

                classified += 1
                if result.is_advice:
                    advice_comments += 1
                    if result.advice_direction:
                        direction_counts[result.advice_direction] = \
                            direction_counts.get(result.advice_direction, 0) + 1

            except Exception as e:
                logger.error(f"Error classifying comment {comment['comment_id']}: {e}")
                errors += 1

            if classified % checkpoint_every == 0:
                logger.info(
                    f"Progress: {classified}/{len(comments)} classified, "
                    f"{advice_comments} advice comments, {errors} errors"
                )

            time.sleep(delay_between_requests)

        logger.info(
            f"Classification complete: {classified} comments classified, "
            f"{advice_comments} advice comments, {errors} errors"
        )
        logger.info(f"Direction breakdown: {direction_counts}")

        return {
            "total_processed": len(comments),
            "classified": classified,
            "advice_comments": advice_comments,
            "errors": errors,
            "direction_counts": direction_counts
        }


def classify_comments(batch_size: Optional[int] = None, **kwargs) -> Dict[str, int]:
    """
    Main entry point for comment classification.

    Args:
        batch_size: Number of comments to classify (None for all)
        **kwargs: Additional arguments passed to classifier

    Returns:
        Results dictionary with counts
    """
    classifier = CommentClassifier()
    return classifier.classify_unclassified_comments(batch_size=batch_size or 50000, **kwargs)


def reclassify_comments(
    batch_size: int = 100,
    delay_between_requests: float = 0.1,
    checkpoint_every: int = 50,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Re-classify already-classified comments with the updated prompt.
    Tracks changes between old and new classifications.

    Args:
        batch_size: Number of comments to re-classify
        delay_between_requests: Delay between API calls
        checkpoint_every: Log progress every N comments
        model: Model to use for classification

    Returns:
        Results dictionary with counts and change tracking
    """
    classifier = CommentClassifier(model=model)
    logger.info(f"Starting re-classification with model {model}")

    comments = db.get_classified_comments(limit=batch_size)
    logger.info(f"Found {len(comments)} comments to re-classify")

    reclassified = 0
    errors = 0

    direction_changes = {
        "unchanged": 0,
        "changed": 0
    }
    change_details = []

    for comment in comments:
        try:
            post_context = comment.get("brief_situation_summary", "")
            if not post_context:
                post_context = f"Title: {comment.get('post_title', 'Unknown')}"

            old_direction = comment.get("old_advice_direction")

            result = classifier.classify_comment(
                comment_body=comment["body"],
                post_context=post_context,
                comment_id=comment["comment_id"]
            )

            # Track changes
            new_direction = result.advice_direction
            if old_direction == new_direction:
                direction_changes["unchanged"] += 1
            else:
                direction_changes["changed"] += 1
                change_details.append({
                    "comment_id": comment["comment_id"],
                    "old": old_direction,
                    "new": new_direction,
                    "gender": comment.get("poster_gender")
                })

            # Store new classification
            db.insert_comment_classification(
                comment_id=comment["comment_id"],
                is_advice=result.is_advice,
                advice_direction=result.advice_direction,
                tone_labels=result.tone_labels,
                classification_model=model
            )

            reclassified += 1

        except Exception as e:
            logger.error(f"Error re-classifying comment {comment['comment_id']}: {e}")
            errors += 1

        if reclassified % checkpoint_every == 0:
            logger.info(
                f"Progress: {reclassified}/{len(comments)}, "
                f"{direction_changes['changed']} changes"
            )

        time.sleep(delay_between_requests)

    # Summarize changes
    change_summary = {}
    for change in change_details:
        key = f"{change['old']} -> {change['new']}"
        if key not in change_summary:
            change_summary[key] = {"total": 0, "male": 0, "female": 0}
        change_summary[key]["total"] += 1
        if change["gender"] in ("male", "female"):
            change_summary[key][change["gender"]] += 1

    logger.info(f"Re-classification complete: {reclassified} comments")
    logger.info(f"Direction changes: {direction_changes}")
    logger.info(f"Change summary: {change_summary}")

    return {
        "total_processed": len(comments),
        "reclassified": reclassified,
        "errors": errors,
        "direction_changes": direction_changes,
        "change_summary": change_summary
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "reclassify":
        batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        results = reclassify_comments(batch_size=batch_size)
    else:
        results = classify_comments(batch_size=10)
    print(json.dumps(results, indent=2))
