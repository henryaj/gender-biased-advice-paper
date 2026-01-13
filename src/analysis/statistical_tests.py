"""Statistical analysis for gender bias research."""

import json
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm

from ..utils.database import db
from ..utils.logging import get_logger

logger = get_logger("statistical_tests")

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def bootstrap_ci(
    data1: np.ndarray,
    data2: np.ndarray,
    stat_func,
    n_bootstrap: int = 1000,
    ci: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data1: First data array
        data2: Second data array
        stat_func: Function that takes two arrays and returns a statistic
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval level

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    point_estimate = stat_func(data1, data2)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)
        bootstrap_stats.append(stat_func(sample1, sample2))

    alpha = 1 - ci
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return point_estimate, lower, upper


class GenderBiasAnalyzer:
    """Analyzer for gender bias in advice comments."""

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.results: Dict[str, Any] = {}

    def load_data(self) -> pd.DataFrame:
        """Load analysis data from database."""
        logger.info("Loading analysis data from database")

        raw_data = db.get_analysis_data()

        if not raw_data:
            raise ValueError("No analysis data available")

        # Convert to DataFrame
        self.data = pd.DataFrame(raw_data)

        # Parse tone_labels JSON
        self.data["tone_labels"] = self.data["tone_labels"].apply(
            lambda x: json.loads(x) if x else []
        )

        logger.info(f"Loaded {len(self.data)} records")
        logger.info(f"Gender distribution:\n{self.data['poster_gender'].value_counts()}")

        return self.data

    def primary_analysis(self) -> Dict[str, Any]:
        """
        Run primary analysis: Do men receive harsher advice?

        Tests:
        - Independent samples t-test
        - Mann-Whitney U test
        - Effect size (Cohen's d)

        Returns:
            Dictionary with test results
        """
        if self.data is None:
            self.load_data()

        logger.info("Running primary analysis")

        results = {
            "research_question": "Do men receive harsher advice than women?",
            "tests": {}
        }

        # Filter to male and female only, with harshness scores
        df = self.data[
            (self.data["poster_gender"].isin(["male", "female"])) &
            (self.data["harshness_score"].notna())
        ].copy()

        if len(df) == 0:
            logger.warning("No data with harshness scores available")
            return {"error": "No harshness scores available"}

        male_scores = df[df["poster_gender"] == "male"]["harshness_score"].values
        female_scores = df[df["poster_gender"] == "female"]["harshness_score"].values

        logger.info(f"Male comments: {len(male_scores)}, Female comments: {len(female_scores)}")

        # Descriptive statistics
        results["descriptive"] = {
            "male": {
                "n": len(male_scores),
                "mean": float(np.mean(male_scores)),
                "std": float(np.std(male_scores)),
                "median": float(np.median(male_scores))
            },
            "female": {
                "n": len(female_scores),
                "mean": float(np.mean(female_scores)),
                "std": float(np.std(female_scores)),
                "median": float(np.median(female_scores))
            }
        }

        # Independent samples t-test
        t_stat, t_pvalue = stats.ttest_ind(male_scores, female_scores)
        results["tests"]["t_test"] = {
            "statistic": float(t_stat),
            "p_value": float(t_pvalue),
            "significant_at_05": t_pvalue < 0.05
        }

        # Mann-Whitney U test (non-parametric alternative)
        u_stat, u_pvalue = stats.mannwhitneyu(male_scores, female_scores, alternative="two-sided")
        results["tests"]["mann_whitney_u"] = {
            "statistic": float(u_stat),
            "p_value": float(u_pvalue),
            "significant_at_05": u_pvalue < 0.05
        }

        # Effect size: Cohen's d
        d = cohens_d(male_scores, female_scores)
        d_est, d_lower, d_upper = bootstrap_ci(male_scores, female_scores, cohens_d)

        results["effect_size"] = {
            "cohens_d": float(d),
            "ci_lower": float(d_lower),
            "ci_upper": float(d_upper),
            "interpretation": self._interpret_cohens_d(d)
        }

        self.results["primary"] = results
        logger.info(f"Primary analysis complete. Cohen's d = {d:.3f}")

        return results

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d value."""
        d_abs = abs(d)
        if d_abs < 0.2:
            size = "negligible"
        elif d_abs < 0.5:
            size = "small"
        elif d_abs < 0.8:
            size = "medium"
        else:
            size = "large"

        direction = "men receive harsher advice" if d > 0 else "women receive harsher advice"
        return f"{size} effect size ({direction})"

    def regression_analysis(self) -> Dict[str, Any]:
        """
        Run regression analysis controlling for confounds.

        Model: harshness ~ gender + relationship_type + source + advice_direction

        Returns:
            Dictionary with regression results
        """
        if self.data is None:
            self.load_data()

        logger.info("Running regression analysis")

        # Filter to usable data
        df = self.data[
            (self.data["poster_gender"].isin(["male", "female"])) &
            (self.data["harshness_score"].notna())
        ].copy()

        if len(df) < 50:
            logger.warning("Insufficient data for regression analysis")
            return {"error": "Insufficient data"}

        # Create binary gender variable (1 = male)
        df["is_male"] = (df["poster_gender"] == "male").astype(int)

        # Run OLS regression with confound controls
        try:
            # Extended formula including situation severity, OP fault, and problem category
            formula = (
                "harshness_score ~ is_male + C(relationship_type) + C(source_name) + "
                "C(advice_direction) + C(situation_severity) + C(op_fault) + C(problem_category)"
            )
            model = ols(formula, data=df).fit()

            results = {
                "model": "OLS Regression",
                "formula": formula,
                "n_observations": int(model.nobs),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "f_statistic": float(model.fvalue) if model.fvalue else None,
                "f_pvalue": float(model.f_pvalue) if model.f_pvalue else None,
                "coefficients": {}
            }

            # Extract key coefficient (is_male)
            for param in model.params.index:
                results["coefficients"][param] = {
                    "estimate": float(model.params[param]),
                    "std_error": float(model.bse[param]),
                    "t_value": float(model.tvalues[param]),
                    "p_value": float(model.pvalues[param]),
                    "significant": model.pvalues[param] < 0.05
                }

            # Highlight the gender effect
            if "is_male" in results["coefficients"]:
                gender_coef = results["coefficients"]["is_male"]
                results["gender_effect"] = {
                    "coefficient": gender_coef["estimate"],
                    "p_value": gender_coef["p_value"],
                    "interpretation": (
                        f"Controlling for other factors, men receive "
                        f"{'harsher' if gender_coef['estimate'] > 0 else 'gentler'} "
                        f"advice by {abs(gender_coef['estimate']):.3f} units "
                        f"(p = {gender_coef['p_value']:.4f})"
                    )
                }

            self.results["regression"] = results
            logger.info(f"Regression complete. R² = {model.rsquared:.3f}")

        except Exception as e:
            logger.error(f"Regression failed: {e}")
            results = {"error": str(e)}

        return results

    def breakdown_by_relationship_type(self) -> Dict[str, Any]:
        """
        Analyze gender bias breakdown by relationship type.

        Returns:
            Dictionary with results by relationship type
        """
        if self.data is None:
            self.load_data()

        logger.info("Running breakdown by relationship type")

        results = {}

        # Filter to usable data
        df = self.data[
            (self.data["poster_gender"].isin(["male", "female"])) &
            (self.data["harshness_score"].notna())
        ].copy()

        for rel_type in df["relationship_type"].unique():
            if pd.isna(rel_type):
                continue

            subset = df[df["relationship_type"] == rel_type]
            male_scores = subset[subset["poster_gender"] == "male"]["harshness_score"].values
            female_scores = subset[subset["poster_gender"] == "female"]["harshness_score"].values

            if len(male_scores) < 5 or len(female_scores) < 5:
                continue

            d = cohens_d(male_scores, female_scores)
            t_stat, t_pvalue = stats.ttest_ind(male_scores, female_scores)

            results[rel_type] = {
                "n_male": len(male_scores),
                "n_female": len(female_scores),
                "mean_male": float(np.mean(male_scores)),
                "mean_female": float(np.mean(female_scores)),
                "cohens_d": float(d),
                "t_statistic": float(t_stat),
                "p_value": float(t_pvalue),
                "significant": t_pvalue < 0.05
            }

        self.results["breakdown_relationship"] = results
        return results

    def breakdown_by_source(self) -> Dict[str, Any]:
        """
        Analyze gender bias breakdown by source (Reddit vs Metafilter).

        Returns:
            Dictionary with results by source
        """
        if self.data is None:
            self.load_data()

        logger.info("Running breakdown by source")

        results = {}

        df = self.data[
            (self.data["poster_gender"].isin(["male", "female"])) &
            (self.data["harshness_score"].notna())
        ].copy()

        for source in df["source_name"].unique():
            subset = df[df["source_name"] == source]
            male_scores = subset[subset["poster_gender"] == "male"]["harshness_score"].values
            female_scores = subset[subset["poster_gender"] == "female"]["harshness_score"].values

            if len(male_scores) < 5 or len(female_scores) < 5:
                continue

            d = cohens_d(male_scores, female_scores)
            t_stat, t_pvalue = stats.ttest_ind(male_scores, female_scores)

            results[source] = {
                "n_male": len(male_scores),
                "n_female": len(female_scores),
                "mean_male": float(np.mean(male_scores)),
                "mean_female": float(np.mean(female_scores)),
                "cohens_d": float(d),
                "t_statistic": float(t_stat),
                "p_value": float(t_pvalue),
                "significant": t_pvalue < 0.05
            }

        self.results["breakdown_source"] = results
        return results

    def breakdown_by_severity(self) -> Dict[str, Any]:
        """
        Analyze gender bias breakdown by situation severity.

        Returns:
            Dictionary with results by severity level
        """
        if self.data is None:
            self.load_data()

        logger.info("Running breakdown by severity")

        results = {}

        df = self.data[
            (self.data["poster_gender"].isin(["male", "female"])) &
            (self.data["harshness_score"].notna())
        ].copy()

        for severity in ["low", "medium", "high"]:
            subset = df[df["situation_severity"] == severity]
            male_scores = subset[subset["poster_gender"] == "male"]["harshness_score"].values
            female_scores = subset[subset["poster_gender"] == "female"]["harshness_score"].values

            if len(male_scores) < 3 or len(female_scores) < 3:
                continue

            d = cohens_d(male_scores, female_scores)
            t_stat, t_pvalue = stats.ttest_ind(male_scores, female_scores)

            results[severity] = {
                "n_male": len(male_scores),
                "n_female": len(female_scores),
                "mean_male": float(np.mean(male_scores)),
                "mean_female": float(np.mean(female_scores)),
                "cohens_d": float(d),
                "t_statistic": float(t_stat),
                "p_value": float(t_pvalue),
                "significant": t_pvalue < 0.05
            }

        self.results["breakdown_severity"] = results
        return results

    def breakdown_by_op_fault(self) -> Dict[str, Any]:
        """
        Analyze gender bias breakdown by OP fault level.

        Returns:
            Dictionary with results by fault level
        """
        if self.data is None:
            self.load_data()

        logger.info("Running breakdown by OP fault")

        results = {}

        df = self.data[
            (self.data["poster_gender"].isin(["male", "female"])) &
            (self.data["harshness_score"].notna())
        ].copy()

        for fault in ["none", "some", "substantial", "unclear"]:
            subset = df[df["op_fault"] == fault]
            male_scores = subset[subset["poster_gender"] == "male"]["harshness_score"].values
            female_scores = subset[subset["poster_gender"] == "female"]["harshness_score"].values

            if len(male_scores) < 3 or len(female_scores) < 3:
                continue

            d = cohens_d(male_scores, female_scores)
            t_stat, t_pvalue = stats.ttest_ind(male_scores, female_scores)

            results[fault] = {
                "n_male": len(male_scores),
                "n_female": len(female_scores),
                "mean_male": float(np.mean(male_scores)),
                "mean_female": float(np.mean(female_scores)),
                "cohens_d": float(d),
                "t_statistic": float(t_stat),
                "p_value": float(t_pvalue),
                "significant": t_pvalue < 0.05
            }

        self.results["breakdown_op_fault"] = results
        return results

    def breakdown_by_problem_category(self) -> Dict[str, Any]:
        """
        Analyze gender bias breakdown by problem category.

        Returns:
            Dictionary with results by category
        """
        if self.data is None:
            self.load_data()

        logger.info("Running breakdown by problem category")

        results = {}

        df = self.data[
            (self.data["poster_gender"].isin(["male", "female"])) &
            (self.data["harshness_score"].notna())
        ].copy()

        for category in df["problem_category"].unique():
            if pd.isna(category):
                continue

            subset = df[df["problem_category"] == category]
            male_scores = subset[subset["poster_gender"] == "male"]["harshness_score"].values
            female_scores = subset[subset["poster_gender"] == "female"]["harshness_score"].values

            if len(male_scores) < 3 or len(female_scores) < 3:
                continue

            d = cohens_d(male_scores, female_scores)
            t_stat, t_pvalue = stats.ttest_ind(male_scores, female_scores)

            results[category] = {
                "n_male": len(male_scores),
                "n_female": len(female_scores),
                "mean_male": float(np.mean(male_scores)),
                "mean_female": float(np.mean(female_scores)),
                "cohens_d": float(d),
                "t_statistic": float(t_stat),
                "p_value": float(t_pvalue),
                "significant": t_pvalue < 0.05
            }

        self.results["breakdown_category"] = results
        return results

    def check_confound_balance(self) -> Dict[str, Any]:
        """
        Check if severity/fault are balanced across genders.

        This is crucial for understanding if observed bias is due to
        men describing objectively worse situations.

        Returns:
            Dictionary with cross-tabulations and chi-square tests
        """
        if self.data is None:
            self.load_data()

        logger.info("Checking confound balance across genders")

        results = {
            "description": "Checks if men and women have different distributions of severity/fault/category",
            "severity_by_gender": {},
            "fault_by_gender": {},
            "category_by_gender": {},
            "chi_square_tests": {}
        }

        df = self.data[self.data["poster_gender"].isin(["male", "female"])].copy()

        # Cross-tabulation: severity by gender
        severity_crosstab = pd.crosstab(df["poster_gender"], df["situation_severity"])
        results["severity_by_gender"] = severity_crosstab.to_dict()

        # Chi-square test for severity independence
        try:
            chi2, pval, dof, expected = stats.chi2_contingency(severity_crosstab)
            results["chi_square_tests"]["severity"] = {
                "chi2": float(chi2),
                "p_value": float(pval),
                "dof": int(dof),
                "significant": pval < 0.05,
                "interpretation": (
                    "Men and women describe situations of DIFFERENT severity"
                    if pval < 0.05 else
                    "Severity is similarly distributed across genders"
                )
            }
        except ValueError:
            results["chi_square_tests"]["severity"] = {"error": "Insufficient data"}

        # Cross-tabulation: fault by gender
        fault_crosstab = pd.crosstab(df["poster_gender"], df["op_fault"])
        results["fault_by_gender"] = fault_crosstab.to_dict()

        # Chi-square test for fault independence
        try:
            chi2, pval, dof, expected = stats.chi2_contingency(fault_crosstab)
            results["chi_square_tests"]["op_fault"] = {
                "chi2": float(chi2),
                "p_value": float(pval),
                "dof": int(dof),
                "significant": pval < 0.05,
                "interpretation": (
                    "Men and women are blamed differently (confound exists!)"
                    if pval < 0.05 else
                    "Fault assignment is similar across genders"
                )
            }
        except ValueError:
            results["chi_square_tests"]["op_fault"] = {"error": "Insufficient data"}

        # Cross-tabulation: category by gender
        cat_crosstab = pd.crosstab(df["poster_gender"], df["problem_category"])
        results["category_by_gender"] = cat_crosstab.to_dict()

        # Chi-square test for category independence
        try:
            chi2, pval, dof, expected = stats.chi2_contingency(cat_crosstab)
            results["chi_square_tests"]["problem_category"] = {
                "chi2": float(chi2),
                "p_value": float(pval),
                "dof": int(dof),
                "significant": pval < 0.05,
                "interpretation": (
                    "Men and women ask about DIFFERENT problem types"
                    if pval < 0.05 else
                    "Problem categories are similar across genders"
                )
            }
        except ValueError:
            results["chi_square_tests"]["problem_category"] = {"error": "Insufficient data"}

        self.results["confound_balance"] = results
        return results

    def tone_label_analysis(self) -> Dict[str, Any]:
        """
        Analyze frequency of tone labels by gender.

        Returns:
            Dictionary with tone label frequencies
        """
        if self.data is None:
            self.load_data()

        logger.info("Running tone label analysis")

        df = self.data[self.data["poster_gender"].isin(["male", "female"])].copy()

        # Count tone labels by gender
        male_counts = defaultdict(int)
        female_counts = defaultdict(int)
        male_total = 0
        female_total = 0

        for _, row in df.iterrows():
            labels = row["tone_labels"]
            if row["poster_gender"] == "male":
                male_total += 1
                for label in labels:
                    male_counts[label] += 1
            else:
                female_total += 1
                for label in labels:
                    female_counts[label] += 1

        # Calculate proportions and run chi-square tests
        results = {
            "male_total": male_total,
            "female_total": female_total,
            "labels": {}
        }

        all_labels = set(male_counts.keys()) | set(female_counts.keys())

        for label in all_labels:
            male_count = male_counts[label]
            female_count = female_counts[label]

            male_prop = male_count / male_total if male_total > 0 else 0
            female_prop = female_count / female_total if female_total > 0 else 0

            # Chi-square test for this label
            contingency = [
                [male_count, male_total - male_count],
                [female_count, female_total - female_count]
            ]

            try:
                chi2, pvalue, dof, expected = stats.chi2_contingency(contingency)
            except ValueError:
                chi2, pvalue = 0, 1.0

            results["labels"][label] = {
                "male_count": male_count,
                "female_count": female_count,
                "male_proportion": float(male_prop),
                "female_proportion": float(female_prop),
                "difference": float(male_prop - female_prop),
                "chi2": float(chi2),
                "p_value": float(pvalue),
                "significant": pvalue < 0.05
            }

        # Sort by difference
        results["labels"] = dict(
            sorted(
                results["labels"].items(),
                key=lambda x: abs(x[1]["difference"]),
                reverse=True
            )
        )

        self.results["tone_labels"] = results
        return results

    def run_all_analyses(self) -> Dict[str, Any]:
        """Run all analyses and return combined results."""
        logger.info("Running all analyses")

        self.load_data()

        all_results = {
            "primary_analysis": self.primary_analysis(),
            "regression": self.regression_analysis(),
            "breakdown_by_relationship_type": self.breakdown_by_relationship_type(),
            "breakdown_by_source": self.breakdown_by_source(),
            "breakdown_by_severity": self.breakdown_by_severity(),
            "breakdown_by_op_fault": self.breakdown_by_op_fault(),
            "breakdown_by_problem_category": self.breakdown_by_problem_category(),
            "confound_balance": self.check_confound_balance(),
            "tone_label_analysis": self.tone_label_analysis()
        }

        # Save results
        output_path = OUTPUT_DIR / "analysis_results.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

        return all_results

    def generate_summary_table(self) -> pd.DataFrame:
        """Generate a summary table of key findings."""
        if not self.results:
            self.run_all_analyses()

        rows = []

        # Primary analysis summary
        if "primary" in self.results:
            primary = self.results["primary"]
            rows.append({
                "Analysis": "Primary (All Data)",
                "Male Mean": primary["descriptive"]["male"]["mean"],
                "Female Mean": primary["descriptive"]["female"]["mean"],
                "Cohen's d": primary["effect_size"]["cohens_d"],
                "p-value": primary["tests"]["t_test"]["p_value"],
                "Significant": primary["tests"]["t_test"]["significant_at_05"]
            })

        # Breakdown summaries
        for key in ["breakdown_relationship", "breakdown_source"]:
            if key in self.results:
                for category, data in self.results[key].items():
                    rows.append({
                        "Analysis": f"{key.replace('breakdown_', '').title()}: {category}",
                        "Male Mean": data["mean_male"],
                        "Female Mean": data["mean_female"],
                        "Cohen's d": data["cohens_d"],
                        "p-value": data["p_value"],
                        "Significant": data["significant"]
                    })

        df = pd.DataFrame(rows)

        # Save to CSV
        output_path = OUTPUT_DIR / "summary_table.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Summary table saved to {output_path}")

        return df


def run_analysis() -> Dict[str, Any]:
    """Main entry point for statistical analysis."""
    analyzer = GenderBiasAnalyzer()
    return analyzer.run_all_analyses()


if __name__ == "__main__":
    results = run_analysis()
    print(json.dumps(results, indent=2))
