"""Reddit scraper using PRAW for r/AmIOverreacting and r/AmItheAsshole."""

import os
import time
import json
from datetime import datetime
from typing import Optional, Generator, Dict, Any
from dataclasses import dataclass

import praw
from praw.models import Submission, Comment

from ..utils.database import db
from ..utils.logging import get_logger, api_logger

logger = get_logger("reddit_scraper")


@dataclass
class RedditConfig:
    """Configuration for Reddit API access."""
    client_id: str
    client_secret: str
    user_agent: str = "GenderBiasResearch/1.0 (Academic Research)"

    @classmethod
    def from_env(cls) -> "RedditConfig":
        """Load configuration from environment variables."""
        client_id = os.environ.get("REDDIT_CLIENT_ID")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET")

        if not client_id or not client_secret:
            raise ValueError(
                "Missing Reddit API credentials. Set REDDIT_CLIENT_ID and "
                "REDDIT_CLIENT_SECRET environment variables."
            )

        return cls(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=os.environ.get(
                "REDDIT_USER_AGENT",
                "GenderBiasResearch/1.0 (Academic Research)"
            )
        )


class RedditScraper:
    """Scraper for Reddit advice subreddits."""

    # Subreddit to source name mapping
    SUBREDDIT_SOURCES = {
        "AmIOverreacting": "amioverreacting",
        "AmItheAsshole": "amitheasshole",
    }

    def __init__(self, config: Optional[RedditConfig] = None):
        """Initialize the scraper with Reddit API credentials."""
        self.config = config or RedditConfig.from_env()
        self.reddit = praw.Reddit(
            client_id=self.config.client_id,
            client_secret=self.config.client_secret,
            user_agent=self.config.user_agent
        )
        logger.info("Reddit scraper initialized")

    def _submission_to_dict(self, submission: Submission) -> Dict[str, Any]:
        """Convert a PRAW submission to a dictionary."""
        return {
            "id": submission.id,
            "title": submission.title,
            "selftext": submission.selftext,
            "author": str(submission.author) if submission.author else "[deleted]",
            "created_utc": submission.created_utc,
            "score": submission.score,
            "upvote_ratio": submission.upvote_ratio,
            "num_comments": submission.num_comments,
            "link_flair_text": submission.link_flair_text,
            "permalink": submission.permalink,
            "url": submission.url,
            "is_self": submission.is_self,
        }

    def _comment_to_dict(self, comment: Comment) -> Dict[str, Any]:
        """Convert a PRAW comment to a dictionary."""
        return {
            "id": comment.id,
            "body": comment.body,
            "author": str(comment.author) if comment.author else "[deleted]",
            "created_utc": comment.created_utc,
            "score": comment.score,
            "parent_id": comment.parent_id,
            "is_submitter": comment.is_submitter,
        }

    def scrape_subreddit(
        self,
        subreddit_name: str,
        limit: int = 5000,
        time_filter: str = "all",
        sort: str = "top",
        min_comments: int = 5,
        checkpoint_every: int = 100
    ) -> Dict[str, int]:
        """
        Scrape posts and comments from a subreddit.

        Args:
            subreddit_name: Name of the subreddit (without r/)
            limit: Maximum number of posts to fetch
            time_filter: Time filter for top posts ('all', 'year', 'month', 'week', 'day')
            sort: Sort method ('top', 'hot', 'new', 'controversial')
            min_comments: Minimum number of comments required to include a post
            checkpoint_every: Log progress every N posts

        Returns:
            Dictionary with counts of posts and comments scraped
        """
        if subreddit_name not in self.SUBREDDIT_SOURCES:
            raise ValueError(f"Unsupported subreddit: {subreddit_name}")

        source_name = self.SUBREDDIT_SOURCES[subreddit_name]
        source_id = db.get_source_id(source_name)

        logger.info(f"Starting scrape of r/{subreddit_name} (limit={limit}, sort={sort})")

        subreddit = self.reddit.subreddit(subreddit_name)

        # Get posts based on sort method
        if sort == "top":
            posts_generator = subreddit.top(time_filter=time_filter, limit=limit)
        elif sort == "hot":
            posts_generator = subreddit.hot(limit=limit)
        elif sort == "new":
            posts_generator = subreddit.new(limit=limit)
        elif sort == "controversial":
            posts_generator = subreddit.controversial(time_filter=time_filter, limit=limit)
        else:
            raise ValueError(f"Unknown sort method: {sort}")

        posts_scraped = 0
        posts_inserted = 0
        comments_scraped = 0
        comments_inserted = 0

        for submission in posts_generator:
            posts_scraped += 1

            # Skip posts with too few comments
            if submission.num_comments < min_comments:
                continue

            # Skip non-text posts
            if not submission.is_self or not submission.selftext:
                continue

            # Skip removed/deleted posts
            if submission.selftext in ("[removed]", "[deleted]"):
                continue

            try:
                # Convert submission to dict for storage
                submission_data = self._submission_to_dict(submission)

                # Insert post
                inserted = db.insert_post(
                    post_id=f"reddit_{submission.id}",
                    source_id=source_id,
                    title=submission.title,
                    body=submission.selftext,
                    author=submission_data["author"],
                    timestamp=datetime.fromtimestamp(submission.created_utc),
                    flair=submission.link_flair_text,
                    raw_json=submission_data
                )

                if inserted:
                    posts_inserted += 1

                    # Fetch and store comments
                    submission.comments.replace_more(limit=0)  # Don't expand "more comments"
                    for comment in submission.comments.list():
                        # Skip deleted/removed comments
                        if comment.body in ("[removed]", "[deleted]"):
                            continue

                        comments_scraped += 1

                        # Determine if top-level
                        is_top_level = comment.parent_id.startswith("t3_")

                        comment_inserted = db.insert_comment(
                            comment_id=f"reddit_{comment.id}",
                            post_id=f"reddit_{submission.id}",
                            body=comment.body,
                            author=str(comment.author) if comment.author else "[deleted]",
                            score=comment.score,
                            timestamp=datetime.fromtimestamp(comment.created_utc),
                            is_top_level=is_top_level
                        )

                        if comment_inserted:
                            comments_inserted += 1

            except Exception as e:
                logger.error(f"Error processing submission {submission.id}: {e}")
                continue

            # Checkpoint logging
            if posts_scraped % checkpoint_every == 0:
                logger.info(
                    f"Progress: {posts_scraped} posts scraped, {posts_inserted} inserted, "
                    f"{comments_inserted} comments inserted"
                )

            # Small delay to be nice to the API
            time.sleep(0.1)

        logger.info(
            f"Completed r/{subreddit_name}: {posts_inserted} posts and "
            f"{comments_inserted} comments inserted"
        )

        return {
            "posts_scraped": posts_scraped,
            "posts_inserted": posts_inserted,
            "comments_scraped": comments_scraped,
            "comments_inserted": comments_inserted
        }

    def scrape_all_subreddits(
        self,
        limit_per_subreddit: int = 5000,
        **kwargs
    ) -> Dict[str, Dict[str, int]]:
        """Scrape all configured subreddits."""
        results = {}

        for subreddit_name in self.SUBREDDIT_SOURCES:
            logger.info(f"Starting scrape of r/{subreddit_name}")
            results[subreddit_name] = self.scrape_subreddit(
                subreddit_name,
                limit=limit_per_subreddit,
                **kwargs
            )

        return results


def scrape_reddit(limit: int = 5000, **kwargs) -> Dict[str, Dict[str, int]]:
    """
    Main entry point for Reddit scraping.

    Args:
        limit: Maximum posts per subreddit
        **kwargs: Additional arguments passed to scrape_subreddit

    Returns:
        Results dictionary with counts by subreddit
    """
    scraper = RedditScraper()
    return scraper.scrape_all_subreddits(limit_per_subreddit=limit, **kwargs)


if __name__ == "__main__":
    # Initialize database first
    from ..utils.database import initialize_database
    initialize_database()

    # Run scraper
    results = scrape_reddit(limit=100)  # Small limit for testing
    print(json.dumps(results, indent=2))
