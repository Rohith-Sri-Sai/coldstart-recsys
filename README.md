# Cold-start recommender

A student project on item cold start: can we recommend new or rarely reviewed
products to users who already have history, without hurting quality on popular
products? Built on Amazon Reviews 2023.

Current state (v0.6): data ingestion, a chronological split with a pre-cutoff
catalog, six models (popularity, TF-IDF content, truncated SVD, LightGCN, a fixed
content/graph hybrid and an adaptive one), equal-budget tuning on dev, bootstrap
intervals and a frozen protocol (`docs/PROTOCOL.md`). Release bundles, a serving
API and deployment come next.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[graph]"   # [graph] adds PyTorch for LightGCN
python -m unittest discover -s tests -v
```

## Pipeline

```bash
recsys-ingest   --reviews data/fixtures/reviews.jsonl --products data/fixtures/products.jsonl --output artifacts/ingestion
recsys-split    --ingestion artifacts/ingestion --output artifacts/splits --train-end-ms 1600000500000 --dev-end-ms 1600001500000 --test-end-ms 1600002500000
recsys-evaluate --splits artifacts/splits --output artifacts/evaluation --phase dev --k 10
```

`recsys-evaluate` with no `--config` runs the three non-graph baselines.
On PowerShell, keep each command on one line. Each step writes to
`runs/<id>/` and only switches the `ACTIVE` pointer once everything has been
written, so a crash halfway leaves the previous run in place.

## How the evaluation works

Windows are `train < train_end <= dev < dev_end <= test < test_end`. For a
window starting at cutoff T, everything is decided using data from before T:

- **Model fit**: every review before T. Dev uses train; test uses train + dev,
  with the settings chosen on dev.
- **Target users**: anyone with a review before T.
- **Catalog**: products whose `Date First Available` is on or before T, or that
  were reviewed before T. Products with neither are left out. A review inside
  the window never makes a product eligible, since that would leak the label.
- **Cohorts**: distinct users who reviewed the product before T.
  0 = cold, 1 to 5 = sparse, 6+ = warm.
- **Seen items**: anything the user reviewed before T. Never recommended,
  never counted as a hit.
- **Metrics**: Recall@K and NDCG@K per cohort, macro-averaged over users with
  at least one label in that cohort. Cold items are ranked against the full
  eligible catalog, not only against other cold items.

Ties are broken with a seeded shuffle, not alphabetically. ASINs are roughly
assigned in sequence, so alphabetical order would quietly favour older products.

Run `--phase test` once, after choosing settings on dev. Every test run is
appended to `artifacts/evaluation/test_runs.jsonl` so repeated runs are visible.

## Models and tuning

| Model | Items nobody has reviewed yet |
|---|---|
| popularity | score 0, random tie order |
| content (TF-IDF of title, description, features, categories) | scored normally |
| svd, lightgcn | ranked last (no collaborative signal) |
| fixed_hybrid: `a*z(content) + (1-a)*z(lightgcn)` | graph part is 0 |
| adaptive_hybrid: graph weight `n/(n+b)`, n = past reviewers of the item | pure content |

```bash
recsys-tune --splits artifacts/splits --output artifacts/tuning --epochs 30
```

This only touches dev. It tries every grid entry in `tune.py` (at most 9 per
model), picks the best NDCG@10, re-runs LightGCN and the hybrids with seeds 0-2,
runs the category ablation and writes to `artifacts/tuning/runs/<id>/`:

- `trials.jsonl`: one line per fitted config (params, metrics, seconds)
- `selected.json`: the chosen configs, used for the final test run
- `dev_report.json`, `summary.md`: dev results with 95% bootstrap intervals and paired differences

The final test run, once, after reading the dev results:

```bash
recsys-evaluate --splits artifacts/splits --output artifacts/evaluation --phase test --config artifacts/tuning/runs/<id>/selected.json
```

LightGCN uses CUDA when available and CPU otherwise (including on a Mac).

### Experiment tracking

`recsys/tracking.py` has a JSONL tracker with a `log(name, params, metrics, tags)`
method. To use MLflow, write a class with the same method and pass it as
`tune(..., tracker=...)`; the docstring shows the four MLflow calls needed.

## Known limitations

- Reviews are not purchases. Low-star reviews count as interactions.
- `Date First Available` is seller-reported and comes from the 2023 crawl. It
  stands in for listing time; I have not verified it. The split manifest counts
  how many products have it and how many have it *after* their first review.
- Product text is also from 2023 and may have changed since the cutoff.
- No stock-out or delisting data: once a product is eligible, it stays eligible.
- Reviews have no event ID, so deduplication uses a hash of user, product,
  time, rating, text and variant ASIN.
- Evaluation scores the full catalog for each user. That is fine for one category
  but too slow for the full dataset.
- LightGCN samples negatives without excluding the user's own positives.

## Smoke test

`scripts/make_synthetic.py` writes fake data in the same format. Use it to check
that everything runs and to time it. Its numbers say nothing about real quality.

## Real data

Download one category (reviews + metadata, raw files rather than the 5-core
subset) from https://amazon-reviews-2023.github.io/ and pass the `.jsonl.gz`
paths to `recsys-ingest`. Choose cutoffs from `timestamp_bounds_ms` in the
ingestion manifest. Example with one-year windows:

```bash
recsys-split --train-end-ms 1609459200000 --dev-end-ms 1640995200000 --test-end-ms 1672531200000   # 2021-01-01 / 2022-01-01 / 2023-01-01 UTC
```

Or run everything up to the test window in one go:

```bash
bash scripts/run_real.sh data/raw/All_Beauty.jsonl.gz data/raw/meta_All_Beauty.jsonl.gz
```

Before trusting any numbers, check `products_with_first_available` and
`dev_labels_dropped_ineligible` in the split manifest. If few products have a
date, the cold cohort will be small and that has to be reported.

## Raw Amazon Reviews 2023 preprocessing (v0.7 addition)

Place the original category files under `data/raw/`:

- `All_Beauty.jsonl.gz` (review events)
- `meta_All_Beauty.jsonl.gz` (product metadata)

Never commit raw data into Git; use DVC after verifying the dataset terms.

Run from the project root after `python -m pip install -e .`:

```powershell
recsys-preprocess --reviews data/raw/All_Beauty.jsonl.gz --products data/raw/meta_All_Beauty.jsonl.gz --output artifacts/real/ingestion
```

Or run `python -m recsys.preprocess` with the same flags. Outputs:

```
artifacts/real/ingestion/
  ACTIVE
  runs/<run-id>/
    records.sqlite       # validated, deduplicated review events and product text
    manifest.json        # source checksums and counters
    quarantine.jsonl     # invalid record locations and rejection reasons
    audit.json           # metadata coverage, availability proxy quality and sparsity
```

The source inputs are never changed. The SQLite database is the input to the existing `recsys-split` command. Inspect `audit.json` before selecting time cutoffs, and note that product metadata is a later snapshot. Historical catalog availability is only a proxy.

This command adds a thin audit layer to the existing streaming ingester; it does **not** replace the temporal splitting step, feature fitting, or training.
