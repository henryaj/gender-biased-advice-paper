# Project Context for Claude

## What This Project Is

This is a research project analyzing **gender bias in online relationship advice**. We scrape posts and comments from:

- **Ask MetaFilter** (relationships tag)
- **Reddit** (r/relationship_advice)

The goal is to determine whether advice-givers respond differently to relationship problems based on the poster's gender.

## Data Pipeline

### 1. Scraping

- **MetaFilter**: `python main.py --scrape` or use `src/scrapers/metafilter_scraper.py`
- **Reddit**: `src/scrapers/reddit_scraper.py` has two methods:
  - `RedditJSONScraper` - no OAuth needed, uses public JSON API
  - `RedditScraper` - requires PRAW/OAuth credentials

Reddit posts are filtered to those with gender markers like `[25M]` or `(30F)` in the title.

### 2. Classification

All classification uses **Claude** via `claude -p` CLI (uses Claude Code subscription, no API key needed).

**Posts** are classified for:
- `poster_gender` - extracted from title markers or LLM-classified
- `is_relationship_advice` - whether it's actually relationship advice
- `situation_severity` - low/medium/high
- `op_fault` - none/some/substantial/unclear
- `problem_category` - communication, infidelity, boundaries, etc.

**Comments** are classified for:
- `is_advice` - whether the comment gives advice
- `advice_direction` - supportive_of_op / critical_of_op / neutral / mixed

### 3. Analysis

Statistical analysis comparing advice direction by poster gender, controlling for confounds.

## Key Scripts

| Script | Purpose |
|--------|---------|
| `main.py` | Main orchestration - run with `--status` to see progress |
| `scripts/classify_reddit.py` | Classification pipeline using `claude -p` CLI |
| `scripts/generate_charts.py` | Generate analysis visualizations |

## Running Classification

```bash
# Check current stats
python scripts/classify_reddit.py --step stats

# Classify Reddit posts (create basic classifications)
python scripts/classify_reddit.py --step posts

# Classify Reddit post details (severity/fault/category)
python scripts/classify_reddit.py --step details --model haiku

# Classify Reddit comments
python scripts/classify_reddit.py --step comments --model sonnet --batch-size 1000

# Reclassify MetaFilter posts
python scripts/classify_reddit.py --step reclassify-mf --model sonnet
```

## Database

SQLite database at `data/research.db`. Key tables:

- `posts` - scraped posts
- `comments` - scraped comments
- `post_classifications` - LLM classifications of posts
- `comment_classifications` - LLM classifications of comments

## Checkpoints

- Reddit scraping saves progress to `data/reddit_checkpoint.json`
- Can resume scraping where it left off
