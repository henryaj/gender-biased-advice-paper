"""Pairwise comparator for generating Bradley-Terry comparison data."""

import os
import json
import time
import random
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

import anthropic

from ..utils.database import db
from ..utils.logging import get_logger, api_logger

logger = get_logger("pairwise_comparator")

# Default model for comparison
DEFAULT_MODEL = "claude-sonnet-4-20250514"

# Comparison prompt template
PAIRWISE_COMPARISON_PROMPT = """Compare these two advice comments for {dimension}.

<comment_a>
{comment_a}
</comment_a>

<comment_b>
{comment_b}
</comment_b>

Which comment is MORE {dimension_description}?

Consider:
- The tone and word choice used
- How the advice is delivered
- The overall emotional impact on the recipient

Return ONLY a JSON object:

{{
  "winner": "a" | "b" | "tie",
  "brief_reasoning": "<1-2 sentence explanation>"
}}"""

DIMENSION_DESCRIPTIONS = {
    "harshness": "harsh (blunt, severe, potentially hurtful, judgmental, critical in tone)",
    "supportiveness": "supportive (validating, encouraging, empathetic, understanding)",
    "constructiveness": "constructive (offering actionable, helpful suggestions)"
}


@dataclass
class ComparisonResult:
    """Result of a pairwise comparison."""
    winner: str  # 'a', 'b', or 'tie'
    reasoning: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComparisonResult":
        winner = data.get("winner", "tie").lower()
        if winner not in ("a", "b", "tie"):
            winner = "tie"
        return cls(
            winner=winner,
            reasoning=data.get("brief_reasoning", "")
        )


class PairwiseComparator:
    """Comparator for generating pairwise comparisons of comment harshness/supportiveness."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the comparator.

        Args:
            model: Claude model to use
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            max_retries: Maximum number of retries on API errors
            retry_delay: Base delay between retries
        """
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Pairwise comparator initialized with model {model}")

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from model response."""
        text = text.strip()

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

    def compare_comments(
        self,
        comment_a: str,
        comment_b: str,
        dimension: str,
        comment_id_a: Optional[str] = None,
        comment_id_b: Optional[str] = None
    ) -> ComparisonResult:
        """
        Compare two comments on a given dimension.

        Args:
            comment_a: First comment text
            comment_b: Second comment text
            dimension: Dimension to compare ('harshness', 'supportiveness', 'constructiveness')
            comment_id_a: Optional ID for logging
            comment_id_b: Optional ID for logging

        Returns:
            ComparisonResult
        """
        if dimension not in DIMENSION_DESCRIPTIONS:
            raise ValueError(f"Unknown dimension: {dimension}")

        # Truncate if needed
        max_length = 2000
        if len(comment_a) > max_length:
            comment_a = comment_a[:max_length] + "..."
        if len(comment_b) > max_length:
            comment_b = comment_b[:max_length] + "..."

        prompt = PAIRWISE_COMPARISON_PROMPT.format(
            dimension=dimension,
            dimension_description=DIMENSION_DESCRIPTIONS[dimension],
            comment_a=comment_a,
            comment_b=comment_b
        )

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )

                response_text = response.content[0].text

                api_logger.log_call(
                    endpoint="messages.create",
                    model=self.model,
                    prompt=prompt,
                    response=response_text,
                    metadata={
                        "comment_id_a": comment_id_a,
                        "comment_id_b": comment_id_b,
                        "dimension": dimension,
                        "classification_type": "pairwise"
                    }
                )

                result_dict = self._parse_json_response(response_text)
                return ComparisonResult.from_dict(result_dict)

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
                return ComparisonResult(winner="tie", reasoning="Parse error")

        return ComparisonResult(winner="tie", reasoning="Max retries exceeded")

    def sample_comment_pairs(
        self,
        num_pairs: int = 500,
        stratify_by_gender: bool = True,
        stratify_by_source: bool = True
    ) -> List[Tuple[Dict, Dict]]:
        """
        Sample comment pairs for pairwise comparison.

        Uses stratified sampling to ensure coverage across:
        - Different poster genders
        - Different sources

        Args:
            num_pairs: Number of pairs to sample
            stratify_by_gender: Ensure balanced sampling across poster genders
            stratify_by_source: Ensure balanced sampling across sources

        Returns:
            List of (comment_a_dict, comment_b_dict) tuples
        """
        logger.info(f"Sampling {num_pairs} comment pairs")

        # Get all classified advice comments
        analysis_data = db.get_analysis_data()

        if len(analysis_data) < 2:
            logger.warning("Not enough classified comments for pairwise comparison")
            return []

        # Group by stratification keys
        groups = defaultdict(list)
        for comment in analysis_data:
            key_parts = []
            if stratify_by_gender:
                key_parts.append(comment.get("poster_gender", "unknown"))
            if stratify_by_source:
                key_parts.append(comment.get("source_name", "unknown"))
            key = tuple(key_parts) if key_parts else ("all",)
            groups[key].append(comment)

        # Sample pairs from each group
        pairs = []
        pairs_per_group = max(1, num_pairs // len(groups))

        for key, comments in groups.items():
            if len(comments) < 2:
                continue

            group_pairs = 0
            attempts = 0
            max_attempts = pairs_per_group * 10

            while group_pairs < pairs_per_group and attempts < max_attempts:
                attempts += 1
                # Sample two different comments
                if len(comments) >= 2:
                    sample = random.sample(comments, 2)
                    # Don't compare comments from the same post
                    if sample[0]["post_id"] != sample[1]["post_id"]:
                        pairs.append((sample[0], sample[1]))
                        group_pairs += 1

        # Shuffle and trim to exact count
        random.shuffle(pairs)
        pairs = pairs[:num_pairs]

        logger.info(f"Sampled {len(pairs)} comment pairs from {len(groups)} groups")
        return pairs

    def run_pairwise_comparisons(
        self,
        dimensions: List[str] = None,
        num_pairs_per_dimension: int = 500,
        delay_between_requests: float = 0.1,
        checkpoint_every: int = 50
    ) -> Dict[str, Any]:
        """
        Run pairwise comparisons for Bradley-Terry scoring.

        Args:
            dimensions: List of dimensions to compare (default: all)
            num_pairs_per_dimension: Number of pairs to compare per dimension
            delay_between_requests: Delay between API calls
            checkpoint_every: Log progress every N comparisons

        Returns:
            Dictionary with counts and results
        """
        if dimensions is None:
            dimensions = list(DIMENSION_DESCRIPTIONS.keys())

        logger.info(f"Starting pairwise comparisons for dimensions: {dimensions}")

        # Sample pairs (same pairs for all dimensions for consistency)
        pairs = self.sample_comment_pairs(num_pairs=num_pairs_per_dimension)

        if not pairs:
            return {"error": "No pairs available for comparison"}

        results = {
            "total_pairs": len(pairs),
            "dimensions": {},
            "errors": 0
        }

        for dimension in dimensions:
            logger.info(f"Processing dimension: {dimension}")
            dim_results = {"a_wins": 0, "b_wins": 0, "ties": 0, "completed": 0}

            for i, (comment_a, comment_b) in enumerate(pairs):
                try:
                    result = self.compare_comments(
                        comment_a=comment_a["comment_body"],
                        comment_b=comment_b["comment_body"],
                        dimension=dimension,
                        comment_id_a=comment_a["comment_id"],
                        comment_id_b=comment_b["comment_id"]
                    )

                    # Store comparison result
                    db.insert_pairwise_comparison(
                        comment_id_a=comment_a["comment_id"],
                        comment_id_b=comment_b["comment_id"],
                        dimension=dimension,
                        winner=result.winner,
                        reasoning=result.reasoning,
                        comparison_model=self.model
                    )

                    # Track counts
                    if result.winner == "a":
                        dim_results["a_wins"] += 1
                    elif result.winner == "b":
                        dim_results["b_wins"] += 1
                    else:
                        dim_results["ties"] += 1
                    dim_results["completed"] += 1

                except Exception as e:
                    logger.error(f"Error in comparison: {e}")
                    results["errors"] += 1

                if (i + 1) % checkpoint_every == 0:
                    logger.info(
                        f"Progress ({dimension}): {i + 1}/{len(pairs)} comparisons, "
                        f"A wins: {dim_results['a_wins']}, B wins: {dim_results['b_wins']}, "
                        f"Ties: {dim_results['ties']}"
                    )

                time.sleep(delay_between_requests)

            results["dimensions"][dimension] = dim_results
            logger.info(f"Completed {dimension}: {dim_results}")

        return results


def run_pairwise_comparisons(
    num_pairs: int = 500,
    dimensions: List[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main entry point for pairwise comparisons.

    Args:
        num_pairs: Number of pairs to compare per dimension
        dimensions: List of dimensions (default: all)
        **kwargs: Additional arguments

    Returns:
        Results dictionary
    """
    comparator = PairwiseComparator()
    return comparator.run_pairwise_comparisons(
        num_pairs_per_dimension=num_pairs,
        dimensions=dimensions,
        **kwargs
    )


if __name__ == "__main__":
    results = run_pairwise_comparisons(num_pairs=10)
    print(json.dumps(results, indent=2))
