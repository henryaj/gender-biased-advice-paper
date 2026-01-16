# Gender Bias in Online Relationship Advice: A Computational Analysis

## Research Question

**Do men receive harsher relationship advice than women when asking similar questions in online forums?**

This study investigates systematic gender bias in advice-giving by analyzing responses to relationship questions on Ask Metafilter, using large language models for classification and statistical methods to control for confounding variables.

---

## Key Finding

**Men are 4.19x more likely to receive critical advice than women** (27.4% vs 8.3%, p < 0.0001), even when controlling for situation severity, poster fault, and problem category.

---

## Materials and Methods

### Data Source

**Ask Metafilter** (https://ask.metafilter.com) - a community Q&A site where users post questions and receive advice from other members.

Ask Metafilter was selected because:
- Consistent HTML structure enabling reliable scraping
- Rich metadata including "favorites" as a quality signal
- Unexplored in academic literature (novel contribution)
- Clear question/answer format ideal for analysis
- Single-page answers (no pagination complexity)

### Data Collection

Posts were scraped from the "relationships" tag on Ask Metafilter using a custom Python scraper with:
- 2.5-second crawl delay (respectful rate limiting)
- Checkpointing for resume capability
- Storage in SQLite database

**Dataset composition:**
- 591 relationship advice posts with identifiable poster gender
- 7,091 total comments
- 6,080 advice-giving comments classified and analyzed

### Classification Pipeline

#### Post Classification

Each post was classified using Claude Haiku to extract:

| Field | Values | Description |
|-------|--------|-------------|
| `poster_gender` | male, female, non-binary, unknown | Inferred from post content |
| `is_relationship_advice` | boolean | Whether post seeks relationship advice |
| `problem_category` | communication, commitment, boundaries, etc. | Type of problem described |
| `situation_severity` | low, medium, high | Severity of situation |
| `op_fault` | none, some, substantial, unclear | Degree to which OP is at fault |

#### Comment Classification

Each comment was classified using **Claude Sonnet 4.5** (`claude-sonnet-4-5-20250929`) to extract:

| Field | Values | Description |
|-------|--------|-------------|
| `is_advice` | boolean | Whether comment gives advice |
| `advice_direction` | supportive_of_op, critical_of_op, neutral, mixed | Stance toward OP |

### Human Validation

To validate classification accuracy, we conducted human spot-checking of **200 randomly sampled comments**. For advice direction—our primary outcome measure—the LLM classifier achieved **96% agreement** with human judgment.

### Statistical Methods

1. **Chi-square tests** for comparing categorical distributions
2. **Odds ratios** for quantifying effect sizes
3. **Stratified analysis** to examine effects within confound strata

---

## Results

### Sample Characteristics

| Gender | Comments | Critical Rate |
|--------|----------|---------------|
| Male | 1,716 | 27.4% |
| Female | 4,364 | 8.3% |

### Primary Finding: Advice Direction

Men receive significantly more critical advice than women.

| Advice Direction | Male (n=1,716) | Female (n=4,364) | Difference |
|-----------------|----------------|------------------|------------|
| Critical of OP | **27.4%** | 8.3% | +19.2 pp |
| Supportive of OP | 29.8% | 42.1% | -12.3 pp |
| Neutral | 31.2% | 38.4% | -7.2 pp |
| Mixed | 11.6% | 11.2% | +0.4 pp |

**Chi-square = 381.80, p < 0.0001**

**Odds ratio:** Men are **4.19x more likely** to receive critical advice compared to women.

### Analysis by Problem Category

| Category | Male Critical | Female Critical | Difference | p-value |
|----------|--------------|-----------------|------------|---------|
| Finances | 44.4% | 8.6% | +35.9 pp | 0.002 |
| Commitment | 33.0% | 5.7% | +27.3 pp | <0.0001 |
| Communication | 29.1% | 7.9% | +21.2 pp | <0.0001 |
| Boundaries | 29.7% | 9.3% | +20.4 pp | <0.0001 |
| Intimacy | 18.5% | 5.6% | +12.9 pp | <0.0001 |
| Lifestyle | 25.7% | 14.3% | +11.4 pp | 0.003 |
| Infidelity | 14.6% | 16.4% | -1.8 pp | 0.886 |

The gender gap is significant across nearly all categories. The exception is **infidelity**, where men and women receive similar rates of critical advice—suggesting that when cheating is involved, commenters apply more uniform standards.

### Confound Analysis

Men and women post about situations with **similar severity and fault distributions** (no significant differences). The gender disparity persists after controlling for these factors.

---

## Discussion

### Summary of Findings

This study provides evidence for substantial gender disparities in online relationship advice:

1. **Large effect size:** 4.19x odds ratio for critical advice
2. **Robust to confounds:** Effects persist after controlling for severity, fault, and problem type
3. **Consistent across categories:** Gender effects observed in 6 of 7 problem categories
4. **High validation:** 96% agreement between LLM and human classification

### Limitations

1. **Single platform:** Results may not generalize beyond Ask Metafilter's community culture
2. **LLM classification:** While validated at 96% accuracy, automated classification introduces some measurement error
3. **Correlation not causation:** We document a pattern but cannot identify its cause
4. **Binary gender:** Analysis focuses on posts with explicitly stated binary gender

---

## Technical Implementation

### Pipeline Architecture

```
Scrape Metafilter → Classify Posts → Filter Dataset → Classify Comments → Statistical Analysis
```

### Key Files

| File | Purpose |
|------|---------|
| `main.py` | CLI orchestration |
| `src/scrapers/metafilter_scraper.py` | Data collection |
| `src/classifiers/post_classifier.py` | Post classification with Claude |
| `src/classifiers/comment_classifier.py` | Comment classification with Claude |
| `scripts/classify_with_claude_code.py` | Batch classification via Claude CLI |
| `scripts/sensitivity_analysis.py` | Statistical analysis |
| `scripts/generate_charts.py` | Chart generation |

### Database Schema

```sql
posts                   -- Questions with metadata
comments                -- Answers with metadata
post_classifications    -- LLM classifications of posts
comment_classifications -- LLM classifications of comments
validation_sample       -- Human validation data
```

### Environment Variables

| Variable | Required For |
|----------|--------------|
| `ANTHROPIC_API_KEY` | LLM classification |

---

## Outputs

- **Academic Paper:** `outputs/paper.pdf`
- **Blog Post:** `outputs/blog_post.md`
- **Analysis Results:** `outputs/analysis_results.json`
- **Charts:**
  - `outputs/chart_critical_by_gender.pdf` - Main finding visualization
  - `outputs/chart_by_category.pdf` - Breakdown by problem category
  - `outputs/chart_advice_distribution.pdf` - Full advice distribution
