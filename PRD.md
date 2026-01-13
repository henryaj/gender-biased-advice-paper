# Gender Bias in Online Relationship Advice

## Product Requirements Document

### Overview

This research project investigates whether men receive more negative, harsh, and critical relationship advice than women in online advice forums. We analyze responses to relationship questions on Ask Metafilter to quantify any systematic gender bias in the advice people receive.

### Research Question

**Do men receive harsher relationship advice than women when asking similar questions in online forums?**

### Data Source

**Ask Metafilter** (https://ask.metafilter.com)

| Tag | Posts | Status |
|-----|-------|--------|
| relationships | ~4,000 | Primary focus |

Ask Metafilter was chosen because:
- Highly scrapeable with consistent HTML structure
- Rich answer data with favorites (quality signal)
- Unexplored in academic literature (novel contribution)
- Single-page answers (no pagination complexity)
- Clear question/answer format ideal for analysis

### Pipeline Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Scrape    │───▶│  Classify   │───▶│   Compare   │
│  Metafilter │    │  with LLM   │    │  Pairwise   │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Visualize  │◀───│   Analyze   │◀───│Bradley-Terry│
│   Results   │    │   Stats     │    │   Scoring   │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Phase 1: Data Collection

**Scraper: `src/scrapers/metafilter_scraper.py`**

Collects from Ask Metafilter:
- Question title, body, author, timestamp
- All answers with author, text, favorites count
- Tags for categorization
- Best answer flags

**Configuration:**
- Crawl delay: 2.5 seconds (respectful rate limiting)
- Checkpointing: Resume-capable after interruption
- Storage: SQLite database

**CLI:**
```bash
python main.py --scrape --limit 4000 --tag relationships
```

### Phase 2: LLM Classification

#### Post Classification (`src/classifiers/post_classifier.py`)

For each question, Claude API determines:

| Field | Values | Description |
|-------|--------|-------------|
| `poster_gender` | male, female, unknown | Inferred from post content |
| `is_relationship_question` | boolean | Filters to relevant posts |
| `relationship_type` | romantic, family, friendship, professional | Category |
| `poster_role` | seeker, giver, observer | Role in situation |

#### Comment Classification (`src/classifiers/comment_classifier.py`)

For each answer, Claude API determines:

| Field | Values | Description |
|-------|--------|-------------|
| `harshness_score` | 1-10 | Overall harshness rating |
| `tone_labels` | [list] | supportive, critical, dismissive, empathetic, etc. |
| `advice_direction` | supportive_of_op, neutral, mixed, critical_of_op | Stance toward OP |
| `blame_assignment` | op_at_fault, other_at_fault, shared, unclear | Who's blamed |

**CLI:**
```bash
python main.py --classify-posts
python main.py --classify-comments
```

### Phase 3: Pairwise Comparisons

**Module: `src/classifiers/pairwise_comparator.py`**

To derive robust harshness scores, we use pairwise comparisons:

1. Select pairs of comments responding to posts of different genders
2. Present both to Claude API, ask which is harsher
3. Record winner/loser pairs for Bradley-Terry model

This approach reduces noise from absolute rating scales and captures relative harshness more reliably.

**CLI:**
```bash
python main.py --pairwise --num-pairs 500
```

### Phase 4: Bradley-Terry Scoring

**Module: `src/analysis/bradley_terry.py`**

The Bradley-Terry model derives latent "harshness" scores from pairwise comparison data:

```
P(comment_i beats comment_j) = score_i / (score_i + score_j)
```

Scores are computed via maximum likelihood estimation, giving each comment a comparable harshness score on a continuous scale.

**CLI:**
```bash
python main.py --bradley-terry
```

### Phase 5: Statistical Analysis

**Module: `src/analysis/statistical_tests.py`**

Statistical tests performed:

| Test | Purpose |
|------|---------|
| Independent t-test | Compare mean harshness by gender |
| Mann-Whitney U | Non-parametric comparison |
| Cohen's d | Effect size measurement |
| Linear regression | Control for confounds (source, relationship type) |
| Bootstrap CI | Confidence intervals on effect sizes |

**Key metrics:**
- Mean harshness difference (male vs female posters)
- Effect size (Cohen's d) with interpretation
- Statistical significance (p-values)
- Breakdown by relationship type and source

**CLI:**
```bash
python main.py --analyze
```

### Phase 6: Visualization

**Module: `src/analysis/visualizations.py`**

Generated figures:
- Harshness score distributions by gender (KDE + boxplot)
- Tone label frequency comparison
- Advice direction breakdown
- Effect sizes with confidence intervals
- Scores by source and gender

**Output:** `outputs/figures/`

**CLI:**
```bash
python main.py --visualize
```

### Database Schema

**SQLite: `data/gender_bias.db`**

```sql
sources          -- Data sources (askmetafilter)
posts            -- Questions with metadata
comments         -- Answers with metadata
post_classifications    -- LLM classifications of posts
comment_classifications -- LLM classifications of comments
pairwise_comparisons    -- Pairwise comparison results
comment_scores          -- Bradley-Terry derived scores
```

### Environment Variables

| Variable | Required For |
|----------|--------------|
| `ANTHROPIC_API_KEY` | LLM classification |

### CLI Reference

```bash
# Full pipeline
python main.py --all

# Individual phases
python main.py --init              # Initialize database
python main.py --scrape            # Scrape Ask Metafilter
python main.py --classify-posts    # Classify posts with LLM
python main.py --classify-comments # Classify comments with LLM
python main.py --pairwise          # Run pairwise comparisons
python main.py --bradley-terry     # Compute BT scores
python main.py --analyze           # Run statistical analysis
python main.py --visualize         # Generate figures

# Options
--limit N              # Max questions to scrape (default: 100)
--tag TAG              # Metafilter tag (default: relationships)
--batch-size N         # Batch size for classification
--num-pairs N          # Number of pairwise comparisons (default: 500)
--skip-scraping        # Skip scraping in full pipeline
--clear-checkpoint     # Clear scraping checkpoint

# Status
python main.py --status            # Show pipeline progress
```

### Expected Outputs

1. **Database:** `data/gender_bias.db` with all collected and classified data
2. **Figures:** `outputs/figures/*.png` with visualizations
3. **Tables:** `outputs/tables/analysis_results.json` with statistical results
4. **Logs:** `logs/` with detailed execution logs

### Success Criteria

1. **Data volume:** 3,000+ classified relationship questions with answers
2. **Classification quality:** >90% agreement on gender inference (spot-check)
3. **Statistical power:** Sufficient sample size to detect medium effect sizes (d > 0.3)
4. **Reproducibility:** Full pipeline runs from scratch with single command

### Timeline Estimates

| Phase | Duration |
|-------|----------|
| Full scrape (4,000 posts) | ~3 hours |
| Post classification | ~2 hours |
| Comment classification | ~8 hours |
| Pairwise comparisons | ~2 hours |
| Analysis & visualization | ~5 minutes |

### Current Status

- [x] Scraper implemented and tested (96 posts, 2,302 answers collected)
- [x] Database schema and utilities complete
- [ ] LLM classification (requires ANTHROPIC_API_KEY)
- [ ] Pairwise comparisons
- [ ] Bradley-Terry scoring
- [ ] Statistical analysis
- [ ] Visualizations
