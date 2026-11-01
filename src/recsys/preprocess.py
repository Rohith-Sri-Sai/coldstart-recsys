"""Raw Amazon Reviews 2023 -> versioned SQLite + human-readable audit.

Does not mutate source files or invent historical product availability.
The ingestion run is promoted first; audit is derived from that immutable run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from .ingest import ingest


def audit_run(output: Path) -> dict:
    root = Path(output)
    run_id = (root / 'ACTIVE').read_text(encoding='utf-8').strip()
    if not run_id or '/' in run_id or '\\' in run_id or run_id.startswith('.'):
        raise ValueError('invalid active run ID')
    run = root / 'runs' / run_id
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    with closing(sqlite3.connect(f'file:{run / "records.sqlite"}?mode=ro', uri=True)) as db:
        def scalar(sql):
            return db.execute(sql).fetchone()[0]

        products = scalar('SELECT COUNT(*) FROM products')
        events = scalar('SELECT COUNT(*) FROM events')
        metadata_events = scalar('SELECT COUNT(*) FROM events e JOIN products p ON p.product_id=e.product_id')
        valid_dates = scalar('SELECT COUNT(*) FROM products WHERE available_from_ms IS NOT NULL')
        earliest = scalar('SELECT MIN(timestamp_ms) FROM events')
        latest = scalar('SELECT MAX(timestamp_ms) FROM events')
        prior_users = dict(db.execute('''
            SELECT bucket, COUNT(*) FROM (
                SELECT CASE WHEN n=1 THEN 'one_review'
                            WHEN n<=5 THEN 'two_to_five_reviews'
                            ELSE 'six_plus_reviews' END AS bucket
                FROM (SELECT user_id, COUNT(*) AS n FROM events GROUP BY user_id)
            ) GROUP BY bucket
        ''').fetchall())
        product_density = dict(db.execute('''
            SELECT bucket, COUNT(*) FROM (
                SELECT CASE WHEN n=1 THEN 'one_review'
                            WHEN n<=5 THEN 'two_to_five_reviews'
                            ELSE 'six_plus_reviews' END AS bucket
                FROM (SELECT product_id, COUNT(*) AS n FROM events GROUP BY product_id)
            ) GROUP BY bucket
        ''').fetchall())
        contradictions = scalar('''
            SELECT COUNT(*) FROM products p JOIN (
                SELECT product_id, MIN(timestamp_ms) AS first_review_ms
                FROM events GROUP BY product_id
            ) e ON e.product_id=p.product_id
            WHERE p.available_from_ms > e.first_review_ms
        ''')
    report = {
        'schema_version': 1,
        'ingestion_run_id': run_id,
        'input_checksums': manifest['inputs'],
        'counts': manifest['counts'],
        'accepted_events': events,
        'products': products,
        'events_with_product_metadata': metadata_events,
        'events_without_product_metadata': events - metadata_events,
        'product_availability_date_coverage': {
            'parsed': valid_dates, 'missing_or_unparsed': products - valid_dates,
            'fraction': (valid_dates / products if products else None),
        },
        'availability_after_first_review_products': contradictions,
        'review_timestamp_bounds_ms': [earliest, latest],
        'users_by_review_count': prior_users,
        'interacted_products_by_review_count': product_density,
        'limitations': [
            'Interaction means review, not a confirmed purchase or click.',
            'Metadata is a later snapshot; title/category text may leak future changes.',
            'Date First Available is a seller-reported proxy, not verified historical listing time.',
            'Unreviewed products cannot be treated as historically eligible without availability evidence.',
        ],
    }
    (run / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def preprocess(reviews: Path, products: Path, output: Path) -> dict:
    ingest(reviews, products, output)
    return audit_run(output)


def main() -> None:
    parser = argparse.ArgumentParser(description='Ingest and audit raw Amazon Reviews 2023 category files')
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--products', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/ingestion'))
    args = parser.parse_args()
    report = preprocess(args.reviews, args.products, args.output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
