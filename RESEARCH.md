# Gender Bias in Online Relationship Advice: A Computational Analysis

## Research Question

**Do men receive harsher relationship advice than women when asking similar questions in online forums?**

This study investigates systematic gender bias in advice-giving by analyzing responses to relationship questions on Ask Metafilter, using large language models for classification and statistical methods to control for confounding variables.

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
- 591 posts scraped
- 526 classified as relationship advice posts
- 13,954 total comments
- 2,174 comments classified for analysis

### Classification Pipeline

#### Post Classification

Each post was classified using Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) to extract:

| Field | Values | Description |
|-------|--------|-------------|
| `poster_gender` | male, female, non-binary, unknown | Inferred from post content |
| `gender_confidence` | 0.0-1.0 | Model confidence in gender inference |
| `is_relationship_advice` | boolean | Whether post seeks relationship advice |
| `relationship_type` | romantic, family, friendship, professional | Category of relationship |
| `problem_category` | communication, commitment, boundaries, etc. | Type of problem described |
| `situation_severity` | low, medium, high | Severity of situation |
| `op_fault` | none, some, substantial, unclear | Degree to which OP is at fault |

#### Comment Classification

Each comment on filtered posts was classified to extract:

| Field | Values | Description |
|-------|--------|-------------|
| `is_advice` | boolean | Whether comment gives advice |
| `advice_direction` | supportive_of_op, critical_of_op, neutral, mixed | Stance toward OP |
| `tone_labels` | [list] | Multiple tone descriptors (see below) |

**Tone labels used:**
- Positive: supportive, encouraging, empathetic, understanding, gentle, constructive
- Negative: judgmental, blaming, condescending, hostile, dismissive, harsh

### Inclusion Criteria

To focus on comparable posts, we applied the following filters:
- `is_relationship_advice = TRUE`
- `problem_category IS NOT NULL` (advice-seeking posts only)
- `poster_gender IN ('male', 'female')` (known binary gender)
- `gender_confidence > 0.7` (high confidence classifications)

This yielded:
- **260 posts** (187 female, 73 male)
- **7,091 comments** eligible for classification
- **2,174 advice-giving comments** analyzed

### Statistical Methods

1. **Chi-square tests** for comparing categorical distributions
2. **Odds ratios** for quantifying effect sizes
3. **Stratified analysis** to examine effects within confound strata
4. **Logistic regression** controlling for severity, fault, and problem category

---

## Results

### Sample Characteristics

| Gender | Posts | Comments | Avg. Confidence |
|--------|-------|----------|-----------------|
| Female | 187 | 1,464 | 91% |
| Male | 73 | 625 | 91% |

### Primary Finding: Advice Direction

Men receive significantly more critical advice and less supportive advice than women.

| Advice Direction | Male | Female |
|-----------------|------|--------|
| Critical of OP | **20.4%** | 5.1% |
| Supportive of OP | 56.8% | **82.2%** |
| Mixed | 9.8% | 3.9% |
| Neutral | 12.8% | 8.9% |

**Chi-square = 186.5, p < 10^-40**

**Odds ratio:** Men are **5.82x more likely** to receive critical (vs supportive) advice compared to women.

### Tone Label Analysis

Highly significant differences in tone labels (all p < 0.001 after Bonferroni correction):

#### Harsh Tones (More Common for Male Posters)

| Tone | Male | Female | Odds Ratio | p-value |
|------|------|--------|------------|---------|
| Hostile | 1.6% | 0.2% | **7.33x** | p < 0.001 |
| Condescending | 9.0% | 1.7% | **5.93x** | p < 10^-14 |
| Blaming | 16.0% | 4.4% | **4.12x** | p < 10^-19 |
| Judgmental | 20.8% | 6.1% | **3.92x** | p < 10^-23 |
| Dismissive | 9.6% | 5.5% | **1.78x** | p < 0.001 |

#### Supportive Tones (More Common for Female Posters)

| Tone | Male | Female | Ratio | p-value |
|------|------|--------|-------|---------|
| Empathetic | 69.3% | **83.1%** | 0.83x | p < 10^-12 |
| Supportive | 59.7% | **75.6%** | 0.79x | p < 10^-13 |
| Encouraging | 51.8% | **69.9%** | 0.74x | p < 10^-15 |

#### No Significant Difference

| Tone | Male | Female | p-value |
|------|------|--------|---------|
| Harsh | 25.3% | 22.7% | p = 0.22 |
| Gentle | 53.3% | 53.6% | p = 0.95 |
| Constructive | 90.6% | 90.3% | p = 0.92 |

### Confound Analysis

Chi-square tests revealed significant differences in confound distributions:

| Confound | Chi-square | p-value | Interpretation |
|----------|-----------|---------|----------------|
| OP Fault | 185.8 | p < 10^-40 | Men classified as more at fault |
| Problem Category | 105.4 | p < 10^-20 | Different problem types by gender |
| Severity | 48.0 | p < 10^-11 | Different severity distributions |

**Key observation:** Men's posts were classified as having higher fault levels:
- "Substantial" fault: Male 7.2% vs Female 1.6%
- "Some" fault: Male 45.8% vs Female 22.0%
- "None": Male 33.6% vs Female 58.3%

### Stratified Analysis

To address confounding, we examined gender effects within strata:

#### Within "No Fault" Posts Only (n=1,109)

| Tone | Male | Female |
|------|------|--------|
| Judgmental | **11.6%** | 6.0% |
| Blaming | **7.2%** | 3.6% |
| Supportive | 71.5% | 75.5% |

#### Within "Low Severity" Posts Only (n=812)

| Tone | Male | Female |
|------|------|--------|
| Judgmental | **13.1%** | 3.5% |
| Blaming | **8.3%** | 2.2% |
| Supportive | 69.7% | 74.9% |

#### Within "Communication" Problem Category (n=905)

| Tone | Male | Female | Difference |
|------|------|--------|------------|
| Judgmental | **21.0%** | 8.1% | +12.9pp |
| Blaming | **17.8%** | 6.4% | +11.4pp |
| Condescending | **11.4%** | 2.1% | +9.3pp |
| Supportive | 60.1% | **76.9%** | -16.8pp |
| Empathetic | 64.1% | **84.1%** | -20.0pp |

**Finding:** Gender effects persist within all strata examined.

### Logistic Regression with Confound Controls

Logistic regression predicting each outcome from gender, controlling for severity, fault, and problem category:

| Outcome | Odds Ratio | 95% CI | p-value |
|---------|-----------|--------|---------|
| Critical advice | **4.57x** | 3.21 - 6.49 | p < 10^-17 |
| Blaming tone | **3.46x** | 2.35 - 5.10 | p < 10^-10 |
| Judgmental tone | **3.34x** | 2.40 - 4.64 | p < 10^-13 |
| Supportive tone | **0.52x** | 0.42 - 0.65 | p < 10^-9 |

**Interpretation:** After controlling for situation severity, OP fault level, and problem category, men remain:
- 4.6x more likely to receive critical advice
- 3.5x more likely to receive blaming comments
- 3.3x more likely to receive judgmental comments
- 1.9x less likely to receive supportive comments

---

## Discussion

### Summary of Findings

This study provides strong evidence for gender bias in online relationship advice:

1. **Large effect sizes:** Odds ratios of 3-7x for harsh tones and critical advice
2. **Robust to confounds:** Effects persist after controlling for severity, fault, and problem type
3. **Consistent across strata:** Gender effects observed within "no fault" posts, "low severity" posts, and specific problem categories
4. **Highly significant:** All key findings significant at p < 10^-9 or better

### Limitations

1. **LLM classification validity:** Both post and comment classifications rely on Claude's judgments. While inter-rater reliability was not formally assessed, the model was instructed with clear criteria.

2. **Single platform:** Results may not generalize beyond Ask Metafilter's community culture.

3. **Partial classification:** Only 2,174 of 7,091 eligible comments were classified due to resource constraints.

4. **Confound measurement:** The confounds (severity, fault, category) were themselves LLM-classified, potentially introducing correlated errors.

5. **Selection bias:** Posts that reveal poster gender may differ systematically from those that don't.

### Future Directions

1. **Human validation:** Manual review of classified comments to assess LLM accuracy
2. **Pairwise comparisons:** Bradley-Terry modeling for continuous harshness scores
3. **Cross-platform replication:** Extend to Reddit, Quora, or other advice forums
4. **Temporal analysis:** Examine whether bias has changed over time
5. **Interaction effects:** Test whether gender effects vary by problem category or severity

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
| `src/analysis/statistical_tests.py` | Statistical analysis |
| `src/analysis/generate_report.py` | Report generation |

### Database Schema

```sql
posts                   -- Questions with metadata
comments                -- Answers with metadata
post_classifications    -- LLM classifications of posts
comment_classifications -- LLM classifications of comments
```

### CLI Reference

```bash
python main.py --scrape --limit 500      # Scrape posts
python main.py --classify-posts          # Classify posts
python main.py --classify-comments       # Classify comments
python main.py --analyze                 # Run statistical analysis
python main.py --generate-report         # Generate HTML report
python main.py --status                  # Show pipeline status
```

### Environment Variables

| Variable | Required For |
|----------|--------------|
| `ANTHROPIC_API_KEY` | LLM classification |

---

## Outputs

- **HTML Report:** `outputs/report.html`
- **Analysis Results:** `outputs/tables/analysis_results.json`
- **Figures:** `outputs/figures/*.png`
- **CSV Tables:** `outputs/tables/*.csv`
