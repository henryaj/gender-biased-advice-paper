#!/usr/bin/env python3
"""
Main orchestration script for the Gender Bias Research Pipeline.

This script runs the complete pipeline from data collection to analysis.
Each phase can be run independently using command-line arguments.

Usage:
    python main.py --all                    # Run complete pipeline
    python main.py --init                   # Initialize database only
    python main.py --scrape                 # Scrape Metafilter data
    python main.py --classify-posts         # Classify posts
    python main.py --classify-comments      # Classify comments
    python main.py --pairwise               # Run pairwise comparisons
    python main.py --bradley-terry          # Compute Bradley-Terry scores
    python main.py --analyze                # Run statistical analysis
    python main.py --visualize              # Generate visualizations
    python main.py --status                 # Show pipeline status
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.database import db, initialize_database
from src.utils.logging import get_logger, LOG_DIR

logger = get_logger("main")


def check_environment():
    """Check that required environment variables are set."""
    import os

    optional_vars = [
        ("ANTHROPIC_API_KEY", "LLM classification"),
    ]

    warnings = []

    for var, purpose in optional_vars:
        if not os.environ.get(var):
            warnings.append(f"  - {var}: Required for {purpose}")

    if warnings:
        logger.warning("Some environment variables are not set:")
        for w in warnings:
            logger.warning(w)
        logger.warning("Set these before running the corresponding pipeline phases.")


def run_init():
    """Initialize the database."""
    logger.info("=" * 60)
    logger.info("PHASE: Database Initialization")
    logger.info("=" * 60)

    initialize_database()

    stats = db.get_summary_stats()
    logger.info(f"Database initialized. Current stats: {json.dumps(stats, indent=2)}")

    return True


def run_scrape_metafilter(max_questions: int = 2000, tag: str = "relationships"):
    """Scrape Metafilter data."""
    logger.info("=" * 60)
    logger.info("PHASE: Metafilter Scraping")
    logger.info(f"Tag: {tag}, Max questions: {max_questions}")
    logger.info("=" * 60)

    from src.scrapers.metafilter_scraper import scrape_metafilter

    results = scrape_metafilter(tag=tag, max_questions=max_questions)
    logger.info(f"Metafilter scraping complete: {json.dumps(results, indent=2)}")

    return results


def clear_metafilter_checkpoint():
    """Clear the Metafilter scraping checkpoint."""
    from src.scrapers.metafilter_scraper import MetafilterScraper
    scraper = MetafilterScraper()
    scraper.clear_checkpoint()
    logger.info("Checkpoint cleared")


def run_classify_posts(batch_size: int = None):
    """Classify posts using Claude API."""
    logger.info("=" * 60)
    logger.info("PHASE: Post Classification")
    logger.info("=" * 60)

    from src.classifiers.post_classifier import classify_posts

    results = classify_posts(batch_size=batch_size)
    logger.info(f"Post classification complete: {json.dumps(results, indent=2)}")

    return results


def run_reclassify_posts(batch_size: int = None):
    """Re-classify all posts with updated prompt (for new confound fields)."""
    logger.info("=" * 60)
    logger.info("PHASE: Post Re-Classification")
    logger.info("Re-classifying to extract situation_severity, op_fault, problem_category")
    logger.info("=" * 60)

    # Run migration first to ensure columns exist
    db.migrate_add_confound_columns()

    from src.classifiers.post_classifier import reclassify_all_posts

    results = reclassify_all_posts(batch_size=batch_size)
    logger.info(f"Post re-classification complete: {json.dumps(results, indent=2)}")

    return results


def run_classify_comments(batch_size: int = None):
    """Classify comments using Claude API."""
    logger.info("=" * 60)
    logger.info("PHASE: Comment Classification")
    logger.info("=" * 60)

    from src.classifiers.comment_classifier import classify_comments

    results = classify_comments(batch_size=batch_size)
    logger.info(f"Comment classification complete: {json.dumps(results, indent=2)}")

    return results


def run_pairwise_comparisons(num_pairs: int = 500):
    """Run pairwise comparisons for Bradley-Terry scoring."""
    logger.info("=" * 60)
    logger.info("PHASE: Pairwise Comparisons")
    logger.info("=" * 60)

    from src.classifiers.pairwise_comparator import run_pairwise_comparisons

    results = run_pairwise_comparisons(num_pairs=num_pairs)
    logger.info(f"Pairwise comparisons complete: {json.dumps(results, indent=2)}")

    return results


def run_bradley_terry():
    """Compute Bradley-Terry scores from pairwise comparisons."""
    logger.info("=" * 60)
    logger.info("PHASE: Bradley-Terry Scoring")
    logger.info("=" * 60)

    from src.analysis.bradley_terry import compute_all_scores, get_score_statistics

    scores = compute_all_scores()

    for dimension in ["harshness", "supportiveness", "constructiveness"]:
        stats = get_score_statistics(dimension)
        logger.info(f"{dimension} statistics: {json.dumps(stats, indent=2)}")

    return scores


def run_analysis():
    """Run statistical analysis."""
    logger.info("=" * 60)
    logger.info("PHASE: Statistical Analysis")
    logger.info("=" * 60)

    from src.analysis.statistical_tests import GenderBiasAnalyzer

    analyzer = GenderBiasAnalyzer()
    results = analyzer.run_all_analyses()

    # Print key findings
    if "primary_analysis" in results:
        primary = results["primary_analysis"]
        if "effect_size" in primary:
            logger.info(f"Key Finding - Cohen's d: {primary['effect_size']['cohens_d']:.3f}")
            logger.info(f"Interpretation: {primary['effect_size']['interpretation']}")

    return results


def run_visualizations(analysis_results=None):
    """Generate visualizations."""
    logger.info("=" * 60)
    logger.info("PHASE: Visualization Generation")
    logger.info("=" * 60)

    from src.analysis.visualizations import generate_all_visualizations

    saved_files = generate_all_visualizations(analysis_results)
    logger.info(f"Generated {len(saved_files)} visualizations")

    return saved_files


def run_generate_report():
    """Generate full analysis report with tables and charts."""
    logger.info("=" * 60)
    logger.info("PHASE: Report Generation")
    logger.info("=" * 60)

    from src.analysis.generate_report import generate_full_report

    results = generate_full_report()
    logger.info(f"Report generated: {results.get('report_path', 'N/A')}")

    return results


def show_status():
    """Show current pipeline status."""
    logger.info("=" * 60)
    logger.info("Pipeline Status")
    logger.info("=" * 60)

    stats = db.get_summary_stats()

    print("\n=== Database Summary ===")
    print(f"Total posts: {stats['total_posts']}")
    print(f"Total comments: {stats['total_comments']}")

    print("\n=== Posts by Source ===")
    for source, count in stats.get('posts_by_source', {}).items():
        print(f"  {source}: {count}")

    print("\n=== Classification Progress ===")
    print(f"Classified posts: {stats['classified_posts']} / {stats['total_posts']}")
    print(f"Classified comments: {stats['classified_comments']} / {stats['total_comments']}")

    print("\n=== Relationship Posts by Gender ===")
    for gender, count in stats.get('relationship_posts_by_gender', {}).items():
        print(f"  {gender}: {count}")

    print("\n=== Pairwise Comparisons ===")
    for dimension, count in stats.get('pairwise_comparisons_by_dimension', {}).items():
        print(f"  {dimension}: {count}")

    return stats


def run_full_pipeline(args):
    """Run the complete pipeline."""
    logger.info("=" * 60)
    logger.info("RUNNING FULL PIPELINE")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    results = {}

    # Phase 1: Initialize
    run_init()

    # Phase 2: Data Collection
    if not args.skip_scraping:
        try:
            results["metafilter"] = run_scrape_metafilter(max_questions=args.limit, tag=args.tag)
        except Exception as e:
            logger.error(f"Metafilter scraping failed: {e}")
            results["metafilter"] = {"error": str(e)}

    # Phase 3: Post Classification
    try:
        results["post_classification"] = run_classify_posts(batch_size=args.batch_size)
    except Exception as e:
        logger.error(f"Post classification failed: {e}")
        results["post_classification"] = {"error": str(e)}

    # Phase 4: Comment Classification
    try:
        results["comment_classification"] = run_classify_comments(batch_size=args.batch_size)
    except Exception as e:
        logger.error(f"Comment classification failed: {e}")
        results["comment_classification"] = {"error": str(e)}

    # Phase 5: Pairwise Comparisons
    try:
        results["pairwise"] = run_pairwise_comparisons(num_pairs=args.num_pairs)
    except Exception as e:
        logger.error(f"Pairwise comparisons failed: {e}")
        results["pairwise"] = {"error": str(e)}

    # Phase 6: Bradley-Terry Scoring
    try:
        results["bradley_terry"] = run_bradley_terry()
    except Exception as e:
        logger.error(f"Bradley-Terry scoring failed: {e}")
        results["bradley_terry"] = {"error": str(e)}

    # Phase 7: Analysis
    try:
        results["analysis"] = run_analysis()
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        results["analysis"] = {"error": str(e)}

    # Phase 8: Visualizations
    try:
        results["visualizations"] = run_visualizations(results.get("analysis"))
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        results["visualizations"] = {"error": str(e)}

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Finished at: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Final status
    show_status()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Gender Bias Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Pipeline phases
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    parser.add_argument("--init", action="store_true", help="Initialize database")
    parser.add_argument("--scrape", action="store_true", help="Scrape Ask Metafilter")
    parser.add_argument("--classify-posts", action="store_true", help="Classify posts")
    parser.add_argument("--reclassify-posts", action="store_true",
                        help="Re-classify all posts with new confound fields")
    parser.add_argument("--classify-comments", action="store_true", help="Classify comments")
    parser.add_argument("--pairwise", action="store_true", help="Run pairwise comparisons")
    parser.add_argument("--bradley-terry", action="store_true", help="Compute BT scores")
    parser.add_argument("--analyze", action="store_true", help="Run analysis")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--generate-report", action="store_true",
                        help="Generate full analysis report with tables and charts")
    parser.add_argument("--status", action="store_true", help="Show pipeline status")

    # Options
    parser.add_argument("--limit", type=int, default=100,
                        help="Max questions to scrape (default: 100)")
    parser.add_argument("--tag", type=str, default="relationships",
                        help="Ask Metafilter tag to scrape (default: relationships)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for classification")
    parser.add_argument("--num-pairs", type=int, default=500,
                        help="Number of pairwise comparisons (default: 500)")
    parser.add_argument("--skip-scraping", action="store_true",
                        help="Skip scraping phase in full pipeline")
    parser.add_argument("--clear-checkpoint", action="store_true",
                        help="Clear scraping checkpoint before starting")

    args = parser.parse_args()

    # Check environment
    check_environment()

    # Run requested phase(s)
    if args.all:
        run_full_pipeline(args)
    elif args.init:
        run_init()
    elif args.scrape:
        run_init()
        if args.clear_checkpoint:
            clear_metafilter_checkpoint()
        run_scrape_metafilter(max_questions=args.limit, tag=args.tag)
    elif args.classify_posts:
        run_classify_posts(batch_size=args.batch_size)
    elif args.reclassify_posts:
        run_reclassify_posts(batch_size=args.batch_size)
    elif args.classify_comments:
        run_classify_comments(batch_size=args.batch_size)
    elif args.pairwise:
        run_pairwise_comparisons(num_pairs=args.num_pairs)
    elif args.bradley_terry:
        run_bradley_terry()
    elif args.analyze:
        run_analysis()
    elif args.visualize:
        # Try to load existing analysis results
        results_path = Path(__file__).parent / "outputs" / "tables" / "analysis_results.json"
        analysis_results = None
        if results_path.exists():
            with open(results_path) as f:
                analysis_results = json.load(f)
        run_visualizations(analysis_results)
    elif args.generate_report:
        run_generate_report()
    elif args.status:
        show_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
