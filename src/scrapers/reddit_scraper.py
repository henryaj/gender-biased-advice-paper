"""Reddit scraper for relationship advice subreddits.

Supports two methods:
1. PRAW (requires OAuth credentials) - for authenticated access
2. JSON API (no credentials) - for public read-only access
"""

import os
import re
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Generator, Dict, Any, List
from dataclasses import dataclass

import requests

# PRAW is optional - only needed for OAuth method
try:
    import praw
    from praw.models import Submission, Comment
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False

from ..utils.database import db
from ..utils.logging import get_logger

logger = get_logger("reddit_scraper")

# Checkpoint file for resumability
CHECKPOINT_FILE = Path(__file__).parent.parent.parent / "data" / "reddit_checkpoint.json"

# Gender pattern: [25M], (30F), 25m, 30f, etc.
GENDER_PATTERN = re.compile(r'[\[\(](\d+)\s*([MmFf])[\]\)]')


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
    """Scraper for Reddit advice subreddits (PRAW-based, requires OAuth)."""

    # Subreddit to source name mapping
    SUBREDDIT_SOURCES = {
        "AmIOverreacting": "amioverreacting",
        "AmItheAsshole": "amitheasshole",
        "relationship_advice": "reddit_relationship_advice",
        "relationships": "reddit_relationships",
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
    Main entry point for Reddit scraping (PRAW-based).

    Args:
        limit: Maximum posts per subreddit
        **kwargs: Additional arguments passed to scrape_subreddit

    Returns:
        Results dictionary with counts by subreddit
    """
    scraper = RedditScraper()
    return scraper.scrape_all_subreddits(limit_per_subreddit=limit, **kwargs)


# =============================================================================
# JSON-based scraper (no OAuth required)
# =============================================================================

class RedditJSONScraper:
    """
    Scraper for Reddit using public JSON API (no credentials needed).

    Uses old.reddit.com which has a simpler interface.
    Rate limited to ~1 request per 2 seconds.
    """

    def __init__(self, delay_between_requests: float = 2.0):
        self.delay = delay_between_requests
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "GenderBiasResearch/1.0 (Academic Research)"
        })
        self.checkpoint = self._load_checkpoint()
        logger.info(f"Reddit JSON scraper initialized (delay={delay_between_requests}s)")

    def _load_checkpoint(self) -> Dict[str, Any]:
        if CHECKPOINT_FILE.exists():
            try:
                with open(CHECKPOINT_FILE) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        return {"scraped_ids": [], "after": None}

    def _save_checkpoint(self):
        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(self.checkpoint, f)

    def _get_json(self, url: str) -> Optional[Dict]:
        """Fetch JSON from Reddit API."""
        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=30)

            if response.status_code == 429:
                wait_time = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                response = self.session.get(url, timeout=30)

            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    @staticmethod
    def extract_gender_markers(text: str) -> List[Dict]:
        """Extract age/gender markers like [25M] or (30F) from text."""
        markers = []
        for match in GENDER_PATTERN.finditer(text):
            age, gender = match.groups()
            markers.append({
                'age': int(age),
                'gender': 'male' if gender.upper() == 'M' else 'female'
            })
        return markers

    def get_posts_with_gender(
        self,
        subreddit: str = "relationship_advice",
        sort: str = "top",
        time_filter: str = "year",
        max_posts: int = 600,
        require_gender_marker: bool = True
    ) -> List[Dict]:
        """Fetch posts, optionally requiring gender markers.

        Args:
            require_gender_marker: If True, only include posts with explicit markers like [25M].
                                   If False, include all posts (gender can be inferred later by LLM).
        """
        posts = []
        after = self.checkpoint.get("after")
        posts_checked = 0

        while len(posts) < max_posts:
            # Use old.reddit.com for simpler API
            url = f"https://old.reddit.com/r/{subreddit}/{sort}.json"
            params = f"?t={time_filter}&limit=100&raw_json=1"
            if after:
                params += f"&after={after}"

            logger.info(f"Fetching posts... (checked {posts_checked}, found {len(posts)})")

            data = self._get_json(url + params)
            if not data:
                break

            children = data.get('data', {}).get('children', [])
            if not children:
                logger.info("No more posts found")
                break

            for child in children:
                post = child.get('data', {})
                post_id = post.get('id')
                posts_checked += 1

                # Skip already scraped
                if post_id in self.checkpoint.get("scraped_ids", []):
                    continue

                # Skip removed/deleted
                selftext = post.get('selftext', '')
                if selftext in ['[removed]', '[deleted]', '']:
                    continue

                # Check for gender markers
                full_text = f"{post.get('title', '')} {selftext}"
                markers = self.extract_gender_markers(full_text)

                # Skip if we require markers and none found
                if require_gender_marker and not markers:
                    continue

                post_data = {
                    'post_id': f"reddit_{post_id}",
                    'reddit_id': post_id,
                    'title': post.get('title', ''),
                    'body': selftext,
                    'author': post.get('author', '[deleted]'),
                    'timestamp': datetime.fromtimestamp(post.get('created_utc', 0)),
                    'score': post.get('score', 0),
                    'num_comments': post.get('num_comments', 0),
                    'url': f"https://reddit.com{post.get('permalink', '')}",
                    'subreddit': subreddit,
                    'gender_marker_present': bool(markers),
                    'all_markers': markers,
                }

                # Add extracted gender if markers present
                if markers:
                    post_data['op_gender'] = markers[0]['gender']
                    post_data['op_age'] = markers[0]['age']
                else:
                    post_data['op_gender'] = None
                    post_data['op_age'] = None

                posts.append(post_data)

                if len(posts) >= max_posts:
                    break

            after = data.get('data', {}).get('after')
            self.checkpoint['after'] = after
            self._save_checkpoint()

            if not after:
                break

        marker_count = sum(1 for p in posts if p['gender_marker_present'])
        logger.info(f"Found {len(posts)} posts ({marker_count} with gender markers, checked {posts_checked} total)")
        return posts

    def get_comments(self, reddit_id: str) -> List[Dict]:
        """Fetch comments for a post."""
        url = f"https://old.reddit.com/comments/{reddit_id}.json?limit=500&depth=10&raw_json=1"

        data = self._get_json(url)
        if not data or len(data) < 2:
            return []

        comments = []

        def extract_comments(children, depth=0):
            for child in children:
                if child.get('kind') != 't1':
                    continue

                comment = child.get('data', {})
                body = comment.get('body', '')

                if body in ['[removed]', '[deleted]', '']:
                    continue

                comments.append({
                    'comment_id': f"reddit_{comment.get('id')}",
                    'body': body,
                    'author': comment.get('author', '[deleted]'),
                    'score': comment.get('score', 0),
                    'timestamp': datetime.fromtimestamp(comment.get('created_utc', 0)),
                    'is_top_level': depth == 0,
                })

                replies = comment.get('replies')
                if replies and isinstance(replies, dict):
                    reply_children = replies.get('data', {}).get('children', [])
                    extract_comments(reply_children, depth + 1)

        comment_listing = data[1].get('data', {}).get('children', [])
        extract_comments(comment_listing)

        return comments

    def scrape(
        self,
        subreddit: str = "relationship_advice",
        max_posts: int = 600,
        min_comments: int = 5,
        checkpoint_every: int = 10,
        time_filter: str = "year",
        sort: str = "top",
        require_gender_marker: bool = True
    ) -> Dict[str, int]:
        """
        Scrape posts and comments from a subreddit.

        Args:
            subreddit: Subreddit to scrape
            max_posts: Target number of posts
            min_comments: Minimum comments required per post
            checkpoint_every: Save checkpoint every N posts
            time_filter: Time filter for posts ('all', 'year', 'month', 'week', 'day')
            sort: Sort method ('top', 'hot', 'new', 'controversial')
            require_gender_marker: If False, include posts without explicit gender markers
        """
        source_id = db.get_source_id(f"reddit_{subreddit}")

        logger.info(f"Starting Reddit scrape of r/{subreddit} (target={max_posts} posts, time={time_filter}, sort={sort}, require_marker={require_gender_marker})")

        posts = self.get_posts_with_gender(
            subreddit=subreddit,
            max_posts=max_posts * 2,  # Fetch extra to account for filtering
            time_filter=time_filter,
            sort=sort,
            require_gender_marker=require_gender_marker
        )

        posts_inserted = 0
        comments_inserted = 0
        skipped = 0

        for i, post in enumerate(posts):
            if posts_inserted >= max_posts:
                break

            if post['num_comments'] < min_comments:
                skipped += 1
                continue

            if post['reddit_id'] in self.checkpoint.get("scraped_ids", []):
                continue

            try:
                comments = self.get_comments(post['reddit_id'])

                if len(comments) < min_comments:
                    skipped += 1
                    continue

                # Build flair - only include gender/age if present
                if post.get('op_gender'):
                    flair = f"gender:{post['op_gender']},age:{post['op_age']}"
                else:
                    flair = "gender:unknown"

                inserted = db.insert_post(
                    post_id=post['post_id'],
                    source_id=source_id,
                    title=post['title'],
                    body=post['body'],
                    author=post['author'],
                    timestamp=post['timestamp'],
                    flair=flair,
                    raw_json={
                        'subreddit': post['subreddit'],
                        'score': post['score'],
                        'url': post['url'],
                        'op_gender': post.get('op_gender'),
                        'op_age': post.get('op_age'),
                        'gender_marker_present': post.get('gender_marker_present', False),
                        'all_markers': post['all_markers'],
                    }
                )

                if inserted:
                    posts_inserted += 1

                    for comment in comments:
                        comment_inserted = db.insert_comment(
                            comment_id=comment['comment_id'],
                            post_id=post['post_id'],
                            body=comment['body'],
                            author=comment['author'],
                            score=comment['score'],
                            timestamp=comment['timestamp'],
                            is_top_level=comment['is_top_level']
                        )
                        if comment_inserted:
                            comments_inserted += 1

                self.checkpoint.setdefault("scraped_ids", []).append(post['reddit_id'])

            except Exception as e:
                logger.error(f"Error processing post {post['post_id']}: {e}")
                continue

            if posts_inserted % checkpoint_every == 0 and posts_inserted > 0:
                self._save_checkpoint()
                logger.info(f"Progress: {posts_inserted}/{max_posts} posts, {comments_inserted} comments")

        self._save_checkpoint()
        logger.info(f"Completed: {posts_inserted} posts, {comments_inserted} comments ({skipped} skipped)")

        return {
            'posts_inserted': posts_inserted,
            'comments_inserted': comments_inserted,
            'skipped': skipped,
        }

    def clear_checkpoint(self):
        """Clear checkpoint to start fresh."""
        self.checkpoint = {"scraped_ids": [], "after": None}
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint cleared")


def scrape_reddit_json(
    subreddit: str = "relationship_advice",
    max_posts: int = 600,
    **kwargs
) -> Dict[str, int]:
    """
    Scrape Reddit using JSON API (no credentials needed).

    Args:
        subreddit: Subreddit to scrape
        max_posts: Target number of posts with gender markers
        **kwargs: Additional arguments

    Returns:
        Results dictionary with counts
    """
    scraper = RedditJSONScraper()
    return scraper.scrape(subreddit=subreddit, max_posts=max_posts, **kwargs)


if __name__ == "__main__":
    import sys

    # Initialize database first
    from ..utils.database import initialize_database
    initialize_database()

    # Parse args
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    subreddit = sys.argv[2] if len(sys.argv) > 2 else "relationship_advice"

    # Use JSON scraper (no credentials needed)
    results = scrape_reddit_json(subreddit=subreddit, max_posts=limit)
    print(json.dumps(results, indent=2))
