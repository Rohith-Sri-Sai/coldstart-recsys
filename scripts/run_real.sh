#!/usr/bin/env bash
# Real-data run up to (not including) the test window.
#
#   bash scripts/run_real.sh data/raw/All_Beauty.jsonl.gz data/raw/meta_All_Beauty.jsonl.gz
#
# Cutoffs default to 2021-01-01 / 2022-01-01 / 2023-01-01 UTC (one-year windows).
# Override with TRAIN_END, DEV_END, TEST_END (Unix ms). Test is NOT run here:
# look at the dev results first, then run the final test once (see README).
set -euo pipefail

REVIEWS=${1:?reviews .jsonl.gz}
META=${2:?metadata .jsonl.gz}
OUT=${OUT:-artifacts/real}
TRAIN_END=${TRAIN_END:-1609459200000}
DEV_END=${DEV_END:-1640995200000}
TEST_END=${TEST_END:-1672531200000}
EPOCHS=${EPOCHS:-30}
mkdir -p "$OUT"

stamp() { date '+%H:%M:%S'; }

echo "$(stamp) ingest"
recsys-ingest --reviews "$REVIEWS" --products "$META" --output "$OUT/ingestion" > "$OUT/ingest_manifest.json"

echo "$(stamp) split"
recsys-split --ingestion "$OUT/ingestion" --output "$OUT/splits" \
  --train-end-ms "$TRAIN_END" --dev-end-ms "$DEV_END" --test-end-ms "$TEST_END" > "$OUT/split_manifest.json"

python - "$OUT" <<'EOF'
import json, sys
out = sys.argv[1]
c = json.load(open(f'{out}/ingest_manifest.json'))['counts']
s = json.load(open(f'{out}/split_manifest.json'))['stats']
print('reviews accepted:', c['reviews_accepted'], '| quarantined:', c['reviews_quarantined'], '| duplicates:', c['reviews_duplicate'])
print('products:', s['products'], '| with Date First Available:', s['products_with_first_available'],
      f"({s['products_with_first_available'] / max(s['products'], 1):.0%})",
      '| date after first review:', s['first_available_after_first_review'])
for phase in ('dev', 'test'):
    print(f'{phase}: labels by cohort', s[f'{phase}_label_cohorts'], '| dropped (not eligible):',
          s[f'{phase}_labels_dropped_ineligible'], '| no metadata:', s[f'{phase}_labels_no_metadata'])
    print(f'{phase}: catalog by cohort', s[f'{phase}_catalog_cohorts'])
EOF

echo "$(stamp) tune on dev (LightGCN epochs=$EPOCHS)"
recsys-tune --splits "$OUT/splits" --output "$OUT/tuning" --epochs "$EPOCHS" > "$OUT/tune_result.json"
RUN="$OUT/tuning/runs/$(cat "$OUT/tuning/ACTIVE")"
echo "$(stamp) done. Dev summary: $RUN/summary.md"
cat "$RUN/summary.md"
python -c "import json;m=json.load(open('$RUN/manifest.json'));print('total seconds:',m['total_seconds'],'| peak RSS MB:',m['peak_rss_mb'])"
