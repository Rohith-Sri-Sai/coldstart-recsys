# Evaluation protocol

Written before any real-data results. If I change it after seeing results, the
change and the reason go in the changelog at the bottom.

## Task

For a user with at least one review before cutoff T, rank the products that are
eligible at T and that the user has not reviewed yet. A hit is a product the user
reviews in the window after T. Data rules (catalog, cohorts, seen items) are in
the README and `split.py`.

## Models

| Name | What it is | Items with no reviews before T |
|---|---|---|
| popularity | review count before T | score 0, random tie order |
| content | TF-IDF of title, description, features (+ categories) | scored normally |
| svd | truncated SVD of the binary user-item matrix | ranked last |
| lightgcn | LightGCN (He et al., 2020) trained with BPR | ranked last |
| fixed_hybrid | `a * z(content) + (1 - a) * z(lightgcn)` | graph term set to 0 |
| adaptive_hybrid | weight on graph `w = n / (n + b)`, n = distinct users who reviewed the item before T | `w = 0`, so content only |

`z()` is a per-user z-score over the candidate items. The adaptive weight is a
simple support-based blend, not a new method; the question is only whether
it beats a single global weight on sparse items.

## Tuning

- Dev window only. Primary metric: NDCG@10 over all evaluated users.
- Each model gets at most 9 configurations (grids in `tune.py`). Popularity has none.
- Hybrids reuse the best content and LightGCN configs and only tune `a` or `b`.
- Selection: highest dev NDCG@10; on a tie, the earlier (simpler) grid entry.
- The chosen configs go to `configs/selected.json` before test is run.
- LightGCN and both hybrids are re-run with seeds 0, 1, 2 at the chosen config.

## Uncertainty

95% bootstrap intervals over users (1000 resamples). Model comparisons use a
paired bootstrap on the same users.

## Hypothesis check (on test, run once)

The adaptive hybrid "wins" only if both hold:

1. Sparse cohort: the paired 95% interval of NDCG@10 (adaptive minus fixed) is above 0.
2. Warm cohort: adaptive NDCG@10 is at least 95% of the fixed hybrid's and of LightGCN's.

The cold cohort is reported but is not evidence for the hypothesis: with no
reviews, the adaptive model is pure content by construction. If the check fails,
that is the result and it gets reported as such.

## Ablations (dev, chosen configs)

- content with and without category text
- content only, graph only, fixed fusion, adaptive fusion

## Changelog

- 2026-10-28: first version.
