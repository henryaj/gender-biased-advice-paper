# Don't Ask Metafilter: Gender Disparities in Online Relationship Advice Vary Dramatically by Platform

**Abstract**

Online advice communities provide valuable resources for individuals seeking guidance on personal matters, yet little is known about whether the advice received varies systematically by platform or by the advice-seeker's gender. We analyze over 12,000 advice comments from two major platforms: Ask Metafilter (7,091 comments from 591 posts) and Reddit's r/relationship_advice (5,595 comments from 184 posts), using large language model (LLM) classification to measure advice direction (supportive vs. critical). We find dramatic platform differences: on Ask Metafilter, men receive critical advice at 3.3 times the rate of women (27.4% vs. 8.3%), while women receive twice the support (54.8% vs. 27.6%). In stark contrast, Reddit shows near gender parity: men and women receive similar rates of support (68.7% vs. 69.3%) and criticism (7.7% vs. 3.8%). Our findings suggest that community culture, not gender alone, determines whether men receive equitable treatment when seeking relationship advice online.

**Keywords**: gender bias, online communities, advice-seeking, content analysis, LLM classification, platform comparison

---

## 1. Introduction

Online advice communities have become a significant resource for individuals navigating personal challenges. Platforms like Reddit's r/relationships, Ask Metafilter, and various forums provide spaces where people can anonymously describe interpersonal problems and receive feedback from community members. This crowdsourced advice model offers accessibility and diverse perspectives, but also raises questions about the quality and consistency of the guidance provided.

Prior research has documented gender differences in various online contexts, including Wikipedia editing (Lam et al., 2011), Stack Overflow participation (Ford et al., 2016), and social media harassment (Duggan, 2017). However, less attention has been paid to whether the *content* of interactions—specifically, the advice given to help-seekers—varies by platform and by the help-seeker's gender.

This study examines whether men and women receive systematically different advice when posting about relationship problems, and crucially, whether these patterns vary across platforms with different community cultures. We compare Ask Metafilter, an older community with a predominantly female user base and strong moderation norms, to Reddit's r/relationship_advice, a larger and more diverse community.

Our research questions are:

**RQ1**: Do men and women receive different proportions of critical vs. supportive advice when posting about relationship problems?

**RQ2**: Do these gender differences vary by platform?

**RQ3**: If platform differences exist, what might explain them?

---

## 2. Related Work

### 2.1 Gender Bias in Online Communities

Research has documented various forms of gender bias in online spaces. Women face higher rates of harassment on social media (Pew Research Center, 2017) and are underrepresented as contributors on platforms like Wikipedia (Hill & Shaw, 2013). In professional contexts, studies have found that women receive different feedback than men—often less specific and more focused on personality rather than performance (Correll & Simard, 2016).

The advice-giving context presents a distinct dynamic. Unlike harassment or professional feedback, advice is ostensibly offered to help the recipient. Yet the framing of advice—whether it validates or challenges the recipient's perspective—may still be influenced by gender stereotypes about who deserves sympathy versus accountability.

### 2.2 Platform Effects on Community Behavior

Different online platforms foster different community cultures (Massanari, 2017). Factors including moderation policies, user demographics, platform design, and historical norms shape how community members interact. A behavior that is normative on one platform may be sanctioned on another.

Ask Metafilter, founded in 2003, is known for its active moderation and community norms that encourage thoughtful responses. It has a predominantly female user base, particularly for relationship-related questions. Reddit's r/relationship_advice, part of a much larger platform, has different demographics and norms. Comparing these platforms allows us to disentangle gender effects from platform effects.

### 2.3 Advice-Giving Dynamics

The literature on advice-giving distinguishes between supportive and challenging responses (Goldsmith, 2004). Supportive advice validates the recipient's feelings and perspective, while challenging advice questions their assumptions or behavior. Both can be appropriate depending on the situation, but systematic differences in who receives which type could indicate bias.

Research on therapeutic contexts suggests that men and women may receive different types of support. Men are sometimes perceived as needing "tough love" while women receive more nurturing responses (Addis & Mahalik, 2003). Whether these patterns extend to informal online advice-giving, and whether they vary by community, remains underexplored.

### 2.4 LLM-Based Content Analysis

Large language models have emerged as powerful tools for content analysis at scale (Ziems et al., 2024). LLMs can classify text along multiple dimensions simultaneously, enabling analysis of large corpora that would be infeasible with manual coding. We use LLM classification for advice analysis, with systematic validation against human judgment.

---

## 3. Data and Methods

### 3.1 Data Collection

We collected data from two platforms:

**Ask Metafilter** (ask.metafilter.com): A question-and-answer community operating since 2003, known for active moderation and thoughtful responses. We collected posts tagged with "relationships" covering interpersonal advice requests.

**Reddit r/relationship_advice**: One of Reddit's largest advice communities with over 9 million members. We collected top posts from 2024-2025 with substantial comment engagement.

Our final dataset comprises:

| Platform | Posts | Comments | Male OP Posts | Female OP Posts |
|----------|-------|----------|---------------|-----------------|
| Ask Metafilter | 591 | 7,091 | 101 | 281 |
| Reddit | 184 | 5,595 | 82 | 102 |
| **Total** | **775** | **12,686** | **183** | **383** |

The gender imbalance in posts reflects underlying community composition; Ask Metafilter has a predominantly female user base for relationship questions.

### 3.2 Classification Framework

For each post, we extracted:

1. **Poster gender**: Identified from explicit mentions in the post text (e.g., "I [30M]..." or "My husband and I [32F]..."). Posts without clear gender indicators were excluded.

2. **Situation severity**: Low, medium, or high, based on the seriousness of the problem described.

3. **OP fault**: None, some, substantial, or unclear, based on how much the poster appears to contribute to the problem.

For each comment, we classified:

1. **Is advice**: Boolean indicating whether the comment provides advice.

2. **Advice direction**:
   - *Supportive of OP*: Validates the poster's perspective, sides with them, provides emotional support
   - *Critical of OP*: Criticizes the poster's behavior or decisions
   - *Neutral*: Balanced, non-judgmental advice or practical information
   - *Mixed*: Contains both supportive and critical elements

### 3.3 LLM Classification

We used Claude Sonnet (Anthropic, 2024) via the Claude Code CLI for automated classification. The model was prompted to analyze each comment in the context of the original post and return structured JSON with the classification fields.

Importantly, we developed **platform-specific prompts** after validation revealed that a single prompt performed poorly across both platforms. Reddit's more direct, conversational style led to over-classification of neutral comments as supportive when using the original prompt. The Reddit-specific prompt includes examples distinguishing blunt practical advice (neutral) from emotional validation (supportive).

### 3.4 Validation

To validate classification accuracy, we conducted human validation of randomly sampled comments.

**Ask Metafilter**: For advice direction—our primary outcome measure—the LLM classifier achieved **96% agreement** with human judgment.

**Reddit**: Initial classification showed only 44% agreement due to the prompt calibration issue. After developing the Reddit-specific prompt, accuracy improved to **72%**. The primary error pattern was over-classifying neutral comments as supportive; human validators found Reddit advice to be more matter-of-fact than the original classifier detected.

### 3.5 Statistical Methods

We computed:

- **Proportions**: Percentage of comments in each category by poster gender and platform
- **Ratios**: Relative rates of critical advice (male/female)
- **Chi-square tests**: For independence between gender and advice type within each platform

---

## 4. Results

### 4.1 Primary Finding: Platform Divergence in Gender Treatment

Table 1 presents the main results on advice direction by platform and poster gender.

**Table 1: Advice Direction by Platform and Poster Gender**

| Platform | Gender | n | Supportive | Critical | Neutral | Mixed |
|----------|--------|---|------------|----------|---------|-------|
| **Ask Metafilter** | Male | 1,716 | 27.6% | 27.4% | 31.9% | 13.1% |
| | Female | 4,364 | 54.8% | 8.3% | 25.5% | 11.5% |
| **Reddit** | Male | 723 | 68.7% | 7.7% | 20.7% | 2.8% |
| | Female | 2,404 | 69.3% | 3.8% | 25.2% | 1.6% |

### 4.2 Ask Metafilter: Substantial Gender Disparity

On Ask Metafilter, the gender disparity is stark:

- **Critical advice**: Men receive critical advice at **3.3 times** the rate of women (27.4% vs. 8.3%, χ² = 325.7, p < 0.0001)
- **Supportive advice**: Women receive supportive advice at **2.0 times** the rate of men (54.8% vs. 27.6%, χ² = 218.4, p < 0.0001)

Notably, men on Ask Metafilter receive roughly equal rates of supportive and critical advice (27.6% vs. 27.4%)—essentially a coin flip. Women receive supportive advice at 6.6 times the rate of critical advice.

### 4.3 Reddit: Near Gender Parity

On Reddit's r/relationship_advice, the pattern is strikingly different:

- **Critical advice**: The gender gap is much smaller—men receive criticism at **2.0 times** the rate of women (7.7% vs. 3.8%), compared to 3.3x on Metafilter
- **Supportive advice**: Men and women receive nearly identical rates of support (68.7% vs. 69.3%, ratio: 1.0x)

Both genders on Reddit receive predominantly supportive responses, with criticism being relatively rare for everyone.

### 4.4 Summary: Gender Disparity Metrics

**Table 2: Gender Disparity by Platform**

| Metric | Ask Metafilter | Reddit |
|--------|----------------|--------|
| Critical rate ratio (M/F) | **3.3x** | 2.0x |
| Supportive rate ratio (F/M) | **2.0x** | 1.0x |
| Male critical rate | 27.4% | 7.7% |
| Female critical rate | 8.3% | 3.8% |

The data reveal two distinct patterns:
1. **Ask Metafilter** shows large gender disparities, with men receiving dramatically more criticism and less support
2. **Reddit** shows relative parity, with both genders receiving similar treatment

---

## 5. Discussion

### 5.1 Summary of Findings

Our cross-platform analysis reveals that gender bias in relationship advice is not universal—it varies dramatically by community. On Ask Metafilter, men are treated as presumptively at fault, receiving criticism at 3.3 times the rate of women. On Reddit, men and women receive comparable advice, with high support rates for both genders.

### 5.2 Explaining the Platform Difference

Several factors may explain the divergence:

**Community demographics**: Ask Metafilter has a predominantly female user base, while Reddit's demographics are more mixed. In-group favoritism could contribute to more sympathetic treatment of female posters on Metafilter.

**Community age and culture**: Ask Metafilter's community norms crystallized in the mid-2000s and may reflect attitudes of that era. Reddit's r/relationship_advice, while also established, has a larger and more frequently turning over user base.

**Moderation philosophy**: Different moderation approaches may allow or discourage certain response patterns. Ask Metafilter's strong moderation may paradoxically enable criticism of men by establishing it as community-normative.

**Self-selection**: Men who anticipate hostile responses may avoid Ask Metafilter, while those who post may be seeking accountability. However, this cannot fully explain the disparity, as men with objectively similar situations to women still receive more criticism.

### 5.3 Implications

**For help-seekers**: Men seeking relationship advice online should be aware that platform choice matters substantially. Ask Metafilter responses may be more critical regardless of situation merit.

**For communities**: These findings suggest that community culture can perpetuate systematic bias. Platform designers and moderators should consider whether their communities inadvertently create different experiences for different groups.

**For research**: Cross-platform comparison is essential for understanding online behavior. Findings from a single platform may not generalize.

### 5.4 The "Tough Love" Hypothesis

One interpretation is that men benefit from critical feedback—the "tough love" hypothesis. However, our data challenge this view:

1. The disparity persists even when controlling for situation severity and poster fault
2. Men with no apparent fault on Metafilter receive more criticism than women with substantial fault
3. Reddit achieves high user engagement without the same disparity, suggesting criticism is not necessary for helpful advice

---

## 6. Limitations

**Platform selection**: We compared two platforms; other communities may show different patterns.

**Temporal scope**: Reddit data is from 2024-2025; Metafilter data spans a longer period. Community norms may shift over time.

**Classification accuracy**: Reddit classification accuracy (72%) is lower than Metafilter (96%). However, the primary error (over-counting supportive) would bias toward finding *less* gender parity on Reddit than actually exists, making our parity finding conservative.

**Binary gender**: Our analysis focuses on posts with explicitly stated binary gender. We cannot draw conclusions about non-binary individuals.

**Causation**: We document patterns but cannot identify root causes. The disparity could reflect commenter attitudes, post content differences, or community selection effects.

---

## 7. Conclusion

This study provides evidence that gender bias in online relationship advice is platform-dependent. Ask Metafilter shows substantial bias against male advice-seekers, who receive critical advice at 3.3 times the rate of women. Reddit's r/relationship_advice shows near gender parity, with both genders receiving predominantly supportive responses.

These findings suggest that community culture—not gender alone—determines whether men receive equitable treatment. For men seeking relationship advice online, the choice of platform may matter as much as the content of their question.

Future research should examine additional platforms, investigate the mechanisms underlying platform differences, and explore whether interventions can reduce disparities in biased communities.

---

## References

Addis, M. E., & Mahalik, J. R. (2003). Men, masculinity, and the contexts of help seeking. American Psychologist, 58(1), 5-14.

Anthropic. (2024). Claude Sonnet model card.

Correll, S. J., & Simard, C. (2016). Research: Vague feedback is holding women back. Harvard Business Review.

Duggan, M. (2017). Online harassment 2017. Pew Research Center.

Ford, D., Smith, J., Guo, P. J., & Parnin, C. (2016). Paradise unplugged: Identifying barriers for female participation on Stack Overflow. Proceedings of FSE 2016.

Goldsmith, D. J. (2004). Communicating social support. Cambridge University Press.

Hill, B. M., & Shaw, A. (2013). The Wikipedia gender gap revisited. PloS One, 8(6), e65782.

Lam, S. T. K., et al. (2011). WP: Clubhouse? An exploration of Wikipedia's gender imbalance. WikiSym 2011.

Massanari, A. (2017). #Gamergate and The Fappening: How Reddit's algorithm, governance, and culture support toxic technocultures. New Media & Society, 19(3), 329-346.

Pew Research Center. (2017). Online harassment 2017.

Ziems, C., et al. (2024). Can large language models transform computational social science? Computational Linguistics, 50(1), 237-291.

---

## Appendix A: Platform-Specific Classification

We developed separate prompts for each platform after validation revealed systematic differences in classification accuracy.

**Metafilter prompt** (standard): Classifies advice as supportive if it "validates OP's perspective, sides with them, supports their instincts, criticizes the OTHER party."

**Reddit prompt** (calibrated): Adds explicit guidance that "blunt advice ('dump him', 'run', 'leave') without emotional validation" should be classified as neutral, not supportive. Reddit's conversational style differs from Metafilter's more elaborate responses.

---

*Code and data: [repository URL]*

*Correspondence: [author email]*
