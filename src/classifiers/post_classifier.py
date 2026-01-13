"""Post classifier using Claude API to identify relationship advice and poster gender."""

import os
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import anthropic

from ..utils.database import db
from ..utils.logging import get_logger, api_logger

logger = get_logger("post_classifier")

# Default model for classification
DEFAULT_MODEL = "claude-3-5-haiku-20241022"

# Classification prompt template
POST_CLASSIFICATION_PROMPT = """Analyze this advice forum post and extract the following information:

<post>
Title: {title}

Body:
{body}
</post>

Please analyze and return a JSON object with these fields:

1. "is_relationship_advice": boolean
   - True if this post is asking for advice about interpersonal relationships (romantic partners, family members, friends, coworkers)
   - False for: legal questions, financial questions, product recommendations, technical questions, etc.

2. "poster_gender": "male" | "female" | "non-binary" | "unknown"
   - Look for explicit statements like "I [24F]", "I (32M)", "my husband", "as a woman", "I'm a guy"
   - Look for contextual clues about the poster's gender
   - Use "unknown" if gender cannot be determined with reasonable confidence

3. "gender_confidence": float between 0.0 and 1.0
   - 1.0 if explicitly stated (e.g., "I [24F]")
   - 0.8-0.9 if strongly implied through relationship context (e.g., "my husband" said by the poster)
   - 0.5-0.7 if weakly implied
   - 0.0 if completely unknown

4. "relationship_type": "romantic" | "family" | "friendship" | "workplace" | "other" | null
   - The primary relationship this advice request concerns
   - null if not relationship advice

5. "situation_summary": string
   - A brief (1-2 sentence) neutral summary of the situation
   - Empty string if not relationship advice

6. "situation_severity": "low" | "medium" | "high" | null
   - "low": Minor disagreement, miscommunication, preference differences, everyday friction
   - "medium": Concerning behavior, relationship strain, trust issues, recurring problems
   - "high": Abuse (physical/emotional/verbal), safety concerns, severe violations like cheating/betrayal, potentially relationship-ending issues
   - null if not relationship advice

7. "op_fault": "none" | "some" | "substantial" | "unclear" | null
   - "none": The poster is clearly the wronged party in this situation
   - "some": The poster contributed to the problem but is not primarily at fault
   - "substantial": The poster is primarily at fault or behaved poorly
   - "unclear": Situation is ambiguous, not enough information, or fault is genuinely shared
   - null if not relationship advice

8. "problem_category": one of the following | null
   - "communication": Issues with expressing needs, listening, understanding each other
   - "infidelity": Cheating, emotional affairs, betrayal of trust
   - "boundaries": Personal space, privacy, autonomy, saying no
   - "commitment": Differing expectations about relationship future, marriage, exclusivity
   - "intimacy": Physical or emotional intimacy issues, affection, sex life
   - "finances": Money disagreements, spending habits, financial responsibility
   - "family_inlaws": Issues with partner's family, in-laws, extended family dynamics
   - "abuse_safety": Physical, emotional, or verbal abuse, controlling behavior, safety concerns
   - "jealousy_trust": Jealousy, suspicion, trust issues (not infidelity-related)
   - "lifestyle": Incompatible lifestyles, habits, values, life goals
   - "other": Does not fit above categories
   - null if not relationship advice

Return ONLY the JSON object, no other text:

{{
  "is_relationship_advice": <boolean>,
  "poster_gender": "<gender>",
  "gender_confidence": <float>,
  "relationship_type": "<type or null>",
  "situation_summary": "<summary>",
  "situation_severity": "<severity or null>",
  "op_fault": "<fault or null>",
  "problem_category": "<category or null>"
}}"""


@dataclass
class PostClassification:
    """Classification result for a post."""
    is_relationship_advice: bool
    poster_gender: str
    gender_confidence: float
    relationship_type: Optional[str]
    situation_summary: str
    situation_severity: Optional[str]
    op_fault: Optional[str]
    problem_category: Optional[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PostClassification":
        return cls(
            is_relationship_advice=data.get("is_relationship_advice", False),
            poster_gender=data.get("poster_gender", "unknown"),
            gender_confidence=data.get("gender_confidence", 0.0),
            relationship_type=data.get("relationship_type"),
            situation_summary=data.get("situation_summary", ""),
            situation_severity=data.get("situation_severity"),
            op_fault=data.get("op_fault"),
            problem_category=data.get("problem_category")
        )


class PostClassifier:
    """Classifier for advice forum posts using Claude API."""

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
        logger.info(f"Post classifier initialized with model {model}")

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from model response, handling common issues."""
        # Try to find JSON in the response
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

        # Try to parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in text
            import re
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise

    def classify_post(
        self,
        title: str,
        body: str,
        post_id: Optional[str] = None
    ) -> PostClassification:
        """
        Classify a single post.

        Args:
            title: Post title
            body: Post body text
            post_id: Optional post ID for logging

        Returns:
            PostClassification result
        """
        # Truncate very long posts
        max_body_length = 8000
        if len(body) > max_body_length:
            body = body[:max_body_length] + "\n\n[truncated]"

        prompt = POST_CLASSIFICATION_PROMPT.format(title=title, body=body)

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )

                response_text = response.content[0].text

                # Log the API call
                api_logger.log_call(
                    endpoint="messages.create",
                    model=self.model,
                    prompt=prompt,
                    response=response_text,
                    metadata={"post_id": post_id, "classification_type": "post"}
                )

                # Parse response
                result = self._parse_json_response(response_text)
                return PostClassification.from_dict(result)

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
                # Return default classification on parse error
                return PostClassification(
                    is_relationship_advice=False,
                    poster_gender="unknown",
                    gender_confidence=0.0,
                    relationship_type=None,
                    situation_summary="",
                    situation_severity=None,
                    op_fault=None,
                    problem_category=None
                )

        # Should not reach here, but return default if we do
        return PostClassification(
            is_relationship_advice=False,
            poster_gender="unknown",
            gender_confidence=0.0,
            relationship_type=None,
            situation_summary="",
            situation_severity=None,
            op_fault=None,
            problem_category=None
        )

    def classify_unclassified_posts(
        self,
        batch_size: int = 100,
        delay_between_requests: float = 0.1,
        checkpoint_every: int = 50
    ) -> Dict[str, int]:
        """
        Classify all posts that haven't been classified yet.

        Args:
            batch_size: Number of posts to process (None for all)
            delay_between_requests: Delay between API calls
            checkpoint_every: Log progress every N posts

        Returns:
            Dictionary with counts
        """
        logger.info("Starting batch post classification")

        posts = db.get_unclassified_posts(limit=batch_size)
        logger.info(f"Found {len(posts)} unclassified posts")

        classified = 0
        relationship_posts = 0
        errors = 0

        for post in posts:
            try:
                result = self.classify_post(
                    title=post["title"] or "",
                    body=post["body"] or "",
                    post_id=post["post_id"]
                )

                # Store classification
                db.insert_post_classification(
                    post_id=post["post_id"],
                    is_relationship_advice=result.is_relationship_advice,
                    poster_gender=result.poster_gender,
                    gender_confidence=result.gender_confidence,
                    relationship_type=result.relationship_type or "",
                    brief_situation_summary=result.situation_summary,
                    classification_model=self.model,
                    situation_severity=result.situation_severity,
                    op_fault=result.op_fault,
                    problem_category=result.problem_category
                )

                classified += 1
                if result.is_relationship_advice:
                    relationship_posts += 1

            except Exception as e:
                logger.error(f"Error classifying post {post['post_id']}: {e}")
                errors += 1

            if classified % checkpoint_every == 0:
                logger.info(
                    f"Progress: {classified}/{len(posts)} classified, "
                    f"{relationship_posts} relationship posts, {errors} errors"
                )

            time.sleep(delay_between_requests)

        logger.info(
            f"Classification complete: {classified} posts classified, "
            f"{relationship_posts} relationship posts, {errors} errors"
        )

        return {
            "total_processed": len(posts),
            "classified": classified,
            "relationship_posts": relationship_posts,
            "errors": errors
        }


def classify_posts(batch_size: Optional[int] = None, **kwargs) -> Dict[str, int]:
    """
    Main entry point for post classification.

    Args:
        batch_size: Number of posts to classify (None for all)
        **kwargs: Additional arguments passed to classifier

    Returns:
        Results dictionary with counts
    """
    classifier = PostClassifier()
    return classifier.classify_unclassified_posts(batch_size=batch_size or 10000, **kwargs)


def reclassify_all_posts(
    batch_size: Optional[int] = None,
    delay_between_requests: float = 0.1,
    **kwargs
) -> Dict[str, int]:
    """
    Re-classify all posts to extract new confound fields.

    Clears existing classifications and re-runs classification on all posts
    to ensure all have the new situation_severity, op_fault, and problem_category.

    Args:
        batch_size: Number of posts to classify (None for all)
        delay_between_requests: Delay between API calls
        **kwargs: Additional arguments passed to classifier

    Returns:
        Results dictionary with counts
    """
    import time
    logger.info("Starting full re-classification of all posts")

    # Get all posts
    with db.get_connection() as conn:
        posts = conn.execute("SELECT * FROM posts").fetchall()
        posts = [dict(row) for row in posts]

    logger.info(f"Found {len(posts)} posts to re-classify")

    if batch_size:
        posts = posts[:batch_size]

    classifier = PostClassifier()
    classified = 0
    relationship_posts = 0
    errors = 0

    for post in posts:
        try:
            result = classifier.classify_post(
                title=post["title"] or "",
                body=post["body"] or "",
                post_id=post["post_id"]
            )

            # Store classification (INSERT OR REPLACE)
            db.insert_post_classification(
                post_id=post["post_id"],
                is_relationship_advice=result.is_relationship_advice,
                poster_gender=result.poster_gender,
                gender_confidence=result.gender_confidence,
                relationship_type=result.relationship_type or "",
                brief_situation_summary=result.situation_summary,
                classification_model=classifier.model,
                situation_severity=result.situation_severity,
                op_fault=result.op_fault,
                problem_category=result.problem_category
            )

            classified += 1
            if result.is_relationship_advice:
                relationship_posts += 1

        except Exception as e:
            logger.error(f"Error classifying post {post['post_id']}: {e}")
            errors += 1

        if classified % 10 == 0:
            logger.info(
                f"Progress: {classified}/{len(posts)} classified, "
                f"{relationship_posts} relationship posts, {errors} errors"
            )

        time.sleep(delay_between_requests)

    logger.info(
        f"Re-classification complete: {classified} posts classified, "
        f"{relationship_posts} relationship posts, {errors} errors"
    )

    return {
        "total_processed": len(posts),
        "classified": classified,
        "relationship_posts": relationship_posts,
        "errors": errors
    }


if __name__ == "__main__":
    results = classify_posts(batch_size=10)
    print(json.dumps(results, indent=2))
