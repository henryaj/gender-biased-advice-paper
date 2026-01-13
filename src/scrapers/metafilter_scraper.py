"""Ask Metafilter scraper for relationship questions."""

import time
import re
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup, NavigableString

from ..utils.database import db
from ..utils.logging import get_logger

logger = get_logger("metafilter_scraper")

BASE_URL = "https://ask.metafilter.com"

# Checkpoint file for resumability
CHECKPOINT_FILE = Path(__file__).parent.parent.parent / "data" / "scrape_checkpoint.json"


class MetafilterScraper:
    """Scraper for Ask Metafilter questions."""

    def __init__(self, delay_between_requests: float = 2.5):
        """
        Initialize the scraper.

        Args:
            delay_between_requests: Seconds to wait between requests
        """
        self.delay = delay_between_requests
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "GenderBiasResearch/1.0 (Academic Research)"
        })
        self.checkpoint = self._load_checkpoint()
        logger.info(f"Metafilter scraper initialized (delay={delay_between_requests}s)")

    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint data for resumability."""
        if CHECKPOINT_FILE.exists():
            try:
                with open(CHECKPOINT_FILE) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        return {"scraped_urls": [], "last_page": 0}

    def _save_checkpoint(self):
        """Save checkpoint data."""
        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(self.checkpoint, f)

    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a page."""
        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse Metafilter date formats."""
        date_str = date_str.strip()

        # Common formats
        formats = [
            "%B %d, %Y %I:%M %p",      # January 5, 2026 9:33 AM
            "%B %d, %Y",                # January 5, 2026
            "%b %d, %Y %I:%M %p",       # Jan 5, 2026 9:33 AM
            "%b %d, %Y",                # Jan 5, 2026
            "%I:%M %p on %B %d",        # 9:33 AM on January 5
            "%I:%M %p on %B %d, %Y",    # 9:33 AM on January 5, 2026
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # Try extracting just date parts
        match = re.search(r'(\w+)\s+(\d+),?\s*(\d{4})?', date_str)
        if match:
            month, day, year = match.groups()
            year = year or str(datetime.now().year)
            try:
                return datetime.strptime(f"{month} {day}, {year}", "%B %d, %Y")
            except ValueError:
                pass

        return None

    def get_question_urls_from_tag(
        self,
        tag: str = "relationships",
        max_pages: int = 100,
        max_questions: Optional[int] = None
    ) -> List[str]:
        """
        Get URLs of questions for a given tag.

        Args:
            tag: Tag to scrape (e.g., 'relationships', 'dating', 'family')
            max_pages: Maximum number of pages to scan
            max_questions: Maximum number of question URLs to return

        Returns:
            List of question URLs
        """
        urls = []
        page = 1

        # Resume from checkpoint if available
        start_page = self.checkpoint.get("last_page", 0) + 1
        if start_page > 1:
            logger.info(f"Resuming from page {start_page}")
            page = start_page

        while page <= max_pages:
            if max_questions and len(urls) >= max_questions:
                break

            tag_url = f"{BASE_URL}/tags/{tag}?page={page}"
            logger.info(f"Fetching tag page {page}: {tag_url}")

            soup = self._get_page(tag_url)
            if not soup:
                break

            # Find all post title links: <h2 class="posttitle"><a href="...">
            post_titles = soup.select("h2.posttitle a")

            if not post_titles:
                logger.info(f"No posts found on page {page}, stopping")
                break

            for link in post_titles:
                href = link.get("href", "")
                if href and re.match(r'https?://ask\.metafilter\.com/\d+/', href):
                    if href not in urls and href not in self.checkpoint.get("scraped_urls", []):
                        urls.append(href)
                        if max_questions and len(urls) >= max_questions:
                            break

            logger.info(f"Page {page}: found {len(post_titles)} posts, total URLs: {len(urls)}")

            # Update checkpoint
            self.checkpoint["last_page"] = page
            self._save_checkpoint()

            # Check if there's a next page
            pagination = soup.select_one("span.smallcopy")
            if pagination:
                total_match = re.search(r'of (\d+)', pagination.get_text())
                if total_match:
                    total = int(total_match.group(1))
                    if page * 50 >= total:
                        logger.info("Reached last page")
                        break

            page += 1

        logger.info(f"Found {len(urls)} question URLs")
        return urls

    def scrape_question(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape a single question and its answers.

        Args:
            url: URL of the question

        Returns:
            Dictionary with question and answer data
        """
        soup = self._get_page(url)
        if not soup:
            return None

        try:
            # Extract post ID from URL
            match = re.search(r'/(\d+)/', url)
            post_id = match.group(1) if match else None
            if not post_id:
                logger.warning(f"Could not extract post ID from {url}")
                return None

            # Get title from h1.posttitle
            title_elem = soup.select_one("h1.posttitle")
            if not title_elem:
                logger.warning(f"Could not find title for {url}")
                return None

            # Title is text before the <br> or <span>
            title_text = ""
            for child in title_elem.children:
                if isinstance(child, NavigableString):
                    title_text += child.strip()
                elif child.name == "br" or child.name == "span":
                    break
            title = title_text.strip()

            # Get date from span.smallcopy inside h1
            date_elem = title_elem.select_one("span.smallcopy")
            post_date = None
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                # Extract date part (before "Subscribe")
                date_match = re.match(r'([\w\s,]+\d{4}\s*\d*:?\d*\s*[APM]*)', date_text)
                if date_match:
                    post_date = self._parse_date(date_match.group(1))

            # Get question body - first div.copy that's not a post listing
            # The question body is in div.copy but after the title
            copy_divs = soup.select("div.copy")
            question_body = ""
            for div in copy_divs:
                # Skip if it's a post listing (has class "post")
                if "post" in div.get("class", []):
                    continue
                # Skip if it contains ads
                if div.select("ins.adsbygoogle"):
                    continue
                # This should be the question body
                text = div.get_text(separator="\n", strip=True)
                if len(text) > 50:  # Reasonable question length
                    question_body = text
                    break

            if not question_body:
                logger.warning(f"Could not find question body for {url}")
                return None

            # Get author from postbyline
            byline = soup.select_one("span.smallcopy.postbyline")
            author = "anonymous"
            if byline:
                author_link = byline.select_one("a")
                if author_link:
                    author = author_link.get_text(strip=True)

            # Get tags
            tags = []
            # Tags are usually in links within the page
            tag_section = soup.find("div", class_="whitesmallcopy")
            if tag_section:
                tag_links = tag_section.select("a[href*='/tags/']")
                tags = [t.get_text(strip=True) for t in tag_links]

            # Get answers (comments)
            answers = []
            comment_divs = soup.select("div.comments")

            for i, comment_div in enumerate(comment_divs):
                comment_id = comment_div.get("id", f"c{i}")
                is_best = "best" in comment_div.get("class", [])

                # Get comment text - everything before the smallcopy span
                comment_text = ""
                for child in comment_div.children:
                    if isinstance(child, NavigableString):
                        comment_text += str(child)
                    elif child.name == "span" and "smallcopy" in child.get("class", []):
                        break
                    elif child.name == "br":
                        comment_text += "\n"
                    elif child.name in ["a", "em", "strong", "i", "b"]:
                        comment_text += child.get_text()
                    elif child.name:
                        comment_text += child.get_text(separator=" ")

                comment_text = comment_text.strip()
                if not comment_text or len(comment_text) < 5:
                    continue

                # Get author and metadata from smallcopy
                meta = comment_div.select_one("span.smallcopy")
                comment_author = "anonymous"
                comment_date = post_date  # Default to post date
                favorites = 0

                if meta:
                    # Author is first link
                    author_link = meta.select_one("a[href*='metafilter.com/user/']")
                    if author_link:
                        comment_author = author_link.get_text(strip=True)

                    # Favorites are in brackets at the end
                    fav_link = meta.select_one("a[href*='/favorited/']")
                    if fav_link:
                        fav_text = fav_link.get("title", "") or fav_link.get_text()
                        fav_match = re.search(r'(\d+)', fav_text)
                        if fav_match:
                            favorites = int(fav_match.group(1))

                    # Try to get timestamp
                    meta_text = meta.get_text()
                    time_match = re.search(r'at\s+(\d+:\d+\s*[APM]+)', meta_text)
                    date_match = re.search(r'on\s+(\w+\s+\d+)', meta_text)
                    if time_match and date_match:
                        full_date = f"{time_match.group(1)} on {date_match.group(1)}"
                        parsed = self._parse_date(full_date)
                        if parsed:
                            comment_date = parsed

                answers.append({
                    "answer_id": f"mf_{post_id}_{comment_id}",
                    "body": comment_text[:10000],
                    "author": comment_author,
                    "favorites": favorites,
                    "is_best": is_best,
                    "timestamp": comment_date
                })

            # Build result dict
            timestamp = post_date or datetime.now()

            # Create a JSON-safe dict for raw_json storage
            raw_json_data = {
                "post_id": f"mf_{post_id}",
                "title": title,
                "body": question_body[:20000],
                "author": author,
                "timestamp": timestamp.isoformat() if timestamp else None,
                "tags": tags,
                "url": url,
                "answers": [
                    {
                        "answer_id": a["answer_id"],
                        "body": a["body"],
                        "author": a["author"],
                        "favorites": a["favorites"],
                        "is_best": a["is_best"],
                        "timestamp": a["timestamp"].isoformat() if a.get("timestamp") else None
                    }
                    for a in answers
                ]
            }

            return {
                "post_id": f"mf_{post_id}",
                "title": title,
                "body": question_body[:20000],
                "author": author,
                "timestamp": timestamp,
                "tags": tags,
                "url": url,
                "answers": answers,
                "raw_json": raw_json_data
            }

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def scrape_tag(
        self,
        tag: str = "relationships",
        max_questions: int = 100,
        checkpoint_every: int = 10
    ) -> Dict[str, int]:
        """
        Scrape questions and answers for a tag.

        Args:
            tag: Tag to scrape
            max_questions: Maximum questions to scrape
            checkpoint_every: Save checkpoint every N questions

        Returns:
            Dictionary with counts
        """
        source_id = db.get_source_id("askmetafilter")

        logger.info(f"Starting scrape of '{tag}' tag (max={max_questions})")

        # Get question URLs
        question_urls = self.get_question_urls_from_tag(
            tag=tag,
            max_questions=max_questions
        )

        posts_scraped = 0
        posts_inserted = 0
        answers_inserted = 0
        errors = 0

        for url in question_urls:
            # Skip if already scraped
            if url in self.checkpoint.get("scraped_urls", []):
                logger.debug(f"Skipping already scraped: {url}")
                continue

            posts_scraped += 1
            data = self.scrape_question(url)

            if not data:
                errors += 1
                continue

            try:
                # Insert post
                inserted = db.insert_post(
                    post_id=data["post_id"],
                    source_id=source_id,
                    title=data["title"],
                    body=data["body"],
                    author=data["author"],
                    timestamp=data["timestamp"],
                    flair=",".join(data["tags"]) if data["tags"] else None,
                    raw_json=data.get("raw_json", {})
                )

                if inserted:
                    posts_inserted += 1

                    # Insert answers as comments
                    for answer in data["answers"]:
                        answer_inserted = db.insert_comment(
                            comment_id=answer["answer_id"],
                            post_id=data["post_id"],
                            body=answer["body"],
                            author=answer["author"],
                            score=answer["favorites"],
                            timestamp=answer.get("timestamp") or data["timestamp"],
                            is_top_level=True
                        )

                        if answer_inserted:
                            answers_inserted += 1

                # Mark as scraped
                self.checkpoint.setdefault("scraped_urls", []).append(url)

            except Exception as e:
                logger.error(f"Error inserting data for {url}: {e}")
                errors += 1
                continue

            if posts_scraped % checkpoint_every == 0:
                self._save_checkpoint()
                logger.info(
                    f"Progress: {posts_scraped}/{len(question_urls)} scraped, "
                    f"{posts_inserted} inserted, {answers_inserted} answers, {errors} errors"
                )

        # Final checkpoint
        self._save_checkpoint()

        logger.info(
            f"Completed: {posts_inserted} questions and "
            f"{answers_inserted} answers inserted ({errors} errors)"
        )

        return {
            "questions_scraped": posts_scraped,
            "questions_inserted": posts_inserted,
            "answers_inserted": answers_inserted,
            "errors": errors
        }

    def clear_checkpoint(self):
        """Clear the checkpoint to start fresh."""
        self.checkpoint = {"scraped_urls": [], "last_page": 0}
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint cleared")


def scrape_metafilter(
    tag: str = "relationships",
    max_questions: int = 100,
    **kwargs
) -> Dict[str, int]:
    """
    Main entry point for Metafilter scraping.

    Args:
        tag: Tag to scrape
        max_questions: Maximum questions to scrape
        **kwargs: Additional arguments

    Returns:
        Results dictionary with counts
    """
    scraper = MetafilterScraper()
    return scraper.scrape_tag(tag=tag, max_questions=max_questions, **kwargs)


if __name__ == "__main__":
    import sys

    # Initialize database first
    from ..utils.database import initialize_database
    initialize_database()

    # Parse args
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    tag = sys.argv[2] if len(sys.argv) > 2 else "relationships"

    # Run scraper
    results = scrape_metafilter(tag=tag, max_questions=limit)
    print(json.dumps(results, indent=2))
