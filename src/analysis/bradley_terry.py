"""Bradley-Terry model for computing latent scores from pairwise comparisons."""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid function

from ..utils.database import db
from ..utils.logging import get_logger

logger = get_logger("bradley_terry")


class BradleyTerryModel:
    """
    Bradley-Terry model for converting pairwise comparisons to latent scores.

    The Bradley-Terry model assumes that each item i has a latent strength s_i,
    and the probability that i beats j is:

    P(i beats j) = s_i / (s_i + s_j) = 1 / (1 + exp(-(log(s_i) - log(s_j))))

    We estimate log-strengths using maximum likelihood.
    """

    def __init__(self, tie_weight: float = 0.5):
        """
        Initialize the model.

        Args:
            tie_weight: Weight given to each item in a tie (0.5 = half win each)
        """
        self.tie_weight = tie_weight
        self.scores: Dict[str, float] = {}
        self.items: List[str] = []
        self.item_to_idx: Dict[str, int] = {}

    def _build_comparison_matrix(
        self,
        comparisons: List[Dict]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build win count matrix from comparisons.

        Args:
            comparisons: List of comparison dicts with 'comment_id_a', 'comment_id_b', 'winner'

        Returns:
            Tuple of (win_matrix, item_list) where win_matrix[i,j] is wins of i over j
        """
        # Collect all unique items
        items: Set[str] = set()
        for comp in comparisons:
            items.add(comp["comment_id_a"])
            items.add(comp["comment_id_b"])

        items = sorted(items)
        self.items = items
        self.item_to_idx = {item: i for i, item in enumerate(items)}
        n = len(items)

        # Build win matrix
        wins = np.zeros((n, n))

        for comp in comparisons:
            i = self.item_to_idx[comp["comment_id_a"]]
            j = self.item_to_idx[comp["comment_id_b"]]
            winner = comp["winner"]

            if winner == "a":
                wins[i, j] += 1
            elif winner == "b":
                wins[j, i] += 1
            else:  # tie
                wins[i, j] += self.tie_weight
                wins[j, i] += self.tie_weight

        return wins, items

    def _negative_log_likelihood(
        self,
        log_scores: np.ndarray,
        wins: np.ndarray
    ) -> float:
        """
        Compute negative log-likelihood for optimization.

        Args:
            log_scores: Log of latent strengths
            wins: Win count matrix

        Returns:
            Negative log-likelihood
        """
        n = len(log_scores)
        nll = 0.0

        for i in range(n):
            for j in range(n):
                if i != j and (wins[i, j] > 0 or wins[j, i] > 0):
                    # P(i beats j) = sigmoid(log_s_i - log_s_j)
                    log_prob_i_beats_j = -np.log1p(np.exp(log_scores[j] - log_scores[i]))
                    log_prob_j_beats_i = -np.log1p(np.exp(log_scores[i] - log_scores[j]))

                    nll -= wins[i, j] * log_prob_i_beats_j
                    nll -= wins[j, i] * log_prob_j_beats_i

        # Add small regularization to prevent extreme values
        nll += 0.001 * np.sum(log_scores ** 2)

        return nll

    def _negative_log_likelihood_gradient(
        self,
        log_scores: np.ndarray,
        wins: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of negative log-likelihood.

        Args:
            log_scores: Log of latent strengths
            wins: Win count matrix

        Returns:
            Gradient array
        """
        n = len(log_scores)
        grad = np.zeros(n)

        for i in range(n):
            for j in range(n):
                if i != j and (wins[i, j] > 0 or wins[j, i] > 0):
                    # Probability that i beats j
                    p_ij = expit(log_scores[i] - log_scores[j])

                    # Gradient contribution
                    # d/d(log_s_i) of -w_ij * log(p_ij) - w_ji * log(1-p_ij)
                    grad[i] += (wins[i, j] + wins[j, i]) * p_ij - wins[i, j]

        # Regularization gradient
        grad += 0.002 * log_scores

        return grad

    def fit(self, comparisons: List[Dict]) -> Dict[str, float]:
        """
        Fit the Bradley-Terry model to pairwise comparisons.

        Args:
            comparisons: List of comparison dicts

        Returns:
            Dictionary mapping item IDs to scores (normalized to 0-1 range)
        """
        if not comparisons:
            logger.warning("No comparisons provided")
            return {}

        logger.info(f"Fitting Bradley-Terry model on {len(comparisons)} comparisons")

        # Build comparison matrix
        wins, items = self._build_comparison_matrix(comparisons)
        n = len(items)

        if n == 0:
            return {}

        logger.info(f"Model has {n} unique items")

        # Initialize log-scores to 0
        log_scores_init = np.zeros(n)

        # Optimize using L-BFGS-B
        result = minimize(
            fun=self._negative_log_likelihood,
            x0=log_scores_init,
            args=(wins,),
            method="L-BFGS-B",
            jac=lambda x, w: self._negative_log_likelihood_gradient(x, w),
            options={"maxiter": 1000, "disp": False}
        )

        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")

        log_scores = result.x

        # Convert to probabilities and normalize to 0-1 range
        # Use min-max normalization on the log scores
        min_score = log_scores.min()
        max_score = log_scores.max()
        range_score = max_score - min_score if max_score > min_score else 1.0

        normalized_scores = (log_scores - min_score) / range_score

        # Store results
        self.scores = {items[i]: float(normalized_scores[i]) for i in range(n)}

        logger.info(
            f"Model fitted. Score range: {normalized_scores.min():.3f} - {normalized_scores.max():.3f}"
        )

        return self.scores

    def get_score(self, item_id: str) -> Optional[float]:
        """Get the score for a specific item."""
        return self.scores.get(item_id)

    def rank_items(self, descending: bool = True) -> List[Tuple[str, float]]:
        """
        Get items ranked by score.

        Args:
            descending: If True, highest scores first

        Returns:
            List of (item_id, score) tuples
        """
        ranked = sorted(self.scores.items(), key=lambda x: x[1], reverse=descending)
        return ranked


def compute_bradley_terry_scores(dimension: str) -> Dict[str, float]:
    """
    Compute Bradley-Terry scores for a dimension.

    Args:
        dimension: The dimension to compute scores for

    Returns:
        Dictionary mapping comment IDs to scores
    """
    logger.info(f"Computing Bradley-Terry scores for dimension: {dimension}")

    # Get comparisons from database
    comparisons = db.get_pairwise_comparisons(dimension)

    if not comparisons:
        logger.warning(f"No comparisons found for dimension: {dimension}")
        return {}

    logger.info(f"Found {len(comparisons)} comparisons")

    # Fit model
    model = BradleyTerryModel()
    scores = model.fit(comparisons)

    # Store scores in database
    score_field = f"{dimension}_score"
    for comment_id, score in scores.items():
        kwargs = {score_field: score}
        db.update_comment_scores(comment_id, **kwargs)

    logger.info(f"Stored {len(scores)} {dimension} scores")

    return scores


def compute_all_scores() -> Dict[str, Dict[str, float]]:
    """
    Compute Bradley-Terry scores for all dimensions.

    Returns:
        Dictionary mapping dimension names to score dictionaries
    """
    dimensions = ["harshness", "supportiveness", "constructiveness"]
    all_scores = {}

    for dimension in dimensions:
        all_scores[dimension] = compute_bradley_terry_scores(dimension)

    return all_scores


def get_score_statistics(dimension: str) -> Dict[str, float]:
    """
    Get statistics about scores for a dimension.

    Args:
        dimension: The dimension to analyze

    Returns:
        Dictionary with statistics
    """
    comparisons = db.get_pairwise_comparisons(dimension)

    if not comparisons:
        return {"error": "No comparisons found"}

    model = BradleyTerryModel()
    scores = model.fit(comparisons)

    if not scores:
        return {"error": "No scores computed"}

    score_values = list(scores.values())

    return {
        "dimension": dimension,
        "n_items": len(scores),
        "n_comparisons": len(comparisons),
        "mean_score": float(np.mean(score_values)),
        "std_score": float(np.std(score_values)),
        "min_score": float(np.min(score_values)),
        "max_score": float(np.max(score_values)),
        "median_score": float(np.median(score_values))
    }


if __name__ == "__main__":
    # Test with sample data
    sample_comparisons = [
        {"comment_id_a": "c1", "comment_id_b": "c2", "winner": "a"},
        {"comment_id_a": "c1", "comment_id_b": "c3", "winner": "a"},
        {"comment_id_a": "c2", "comment_id_b": "c3", "winner": "a"},
        {"comment_id_a": "c2", "comment_id_b": "c4", "winner": "b"},
        {"comment_id_a": "c3", "comment_id_b": "c4", "winner": "tie"},
    ]

    model = BradleyTerryModel()
    scores = model.fit(sample_comparisons)
    print("Sample scores:", json.dumps(scores, indent=2))
    print("Ranked:", model.rank_items())
