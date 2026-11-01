"""Streaming, idempotent-ish source ingestion with an atomic run pointer.

Raw Amazon Reviews 2023 JSONL(.gz) is never modified. For reproducibility,
input paths and content are recorded as hashes. Reviews do NOT have event IDs:
we use a deterministic content fingerprint, which may merge truly distinct
reviews with identical raw identity fields; this is an explicit limitation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone


def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError('not_an_object')
                yield line_no, value, None
            except (json.JSONDecodeError, ValueError) as exc:
                yield line_no, None, f'json:{type(exc).__name__}'


def nonempty_str(value):
    return isinstance(value, str) and bool(value.strip())


def valid_review(r):
    if not nonempty_str(r.get('user_id')) or not nonempty_str(r.get('parent_asin')):
        return 'missing_user_or_item'
    t = r.get('timestamp')
    if type(t) is not int or not 0 < t <= 4102444800000:
        return 'invalid_timestamp_ms'
    v = r.get('rating')
    if type(v) not in (float, int) or not 1 <= v <= 5:
        return 'invalid_rating'
    return None


def valid_product(p):
    if not nonempty_str(p.get('parent_asin')):
        return 'missing_product_id'
    if not nonempty_str(p.get('title')):
        return 'missing_product_title'
    return None


DATE_FORMATS = ('%B %d, %Y', '%b %d, %Y')


def first_available_ms(details):
    """Parse details['Date First Available'] (day precision, UTC midnight).

    This is seller-reported and comes from the 2023 crawl, so it is a proxy for
    listing time, not a verified one. Returns None when missing or unparseable.
    """
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            return None
    if not isinstance(details, dict):
        return None
    raw = details.get('Date First Available')
    if not isinstance(raw, str):
        return None
    for fmt in DATE_FORMATS:
        try:
            day = datetime.strptime(raw.strip(), fmt).replace(tzinfo=timezone.utc)
            return int(day.timestamp() * 1000)
        except ValueError:
            pass
    return None


def fingerprint(r):
    # No stable event ID provided by Amazon; document this assumption.
    fields = [r['user_id'], r['parent_asin'], r['timestamp'], r['rating'], r.get('text'), r.get('asin')]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ingest(reviews: Path, products: Path, output: Path, *, fail_before_commit=False):
    reviews, products, output = Path(reviews), Path(products), Path(output)
    for path in (reviews, products):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    runs = output / 'runs'
    runs.mkdir(exist_ok=True)
    run_id = str(uuid.uuid4())
    stage = runs / ('.building-' + run_id)
    stage.mkdir()
    db_path = stage / 'records.sqlite'
    conn = sqlite3.connect(db_path)
    counters = {k: 0 for k in ('reviews_read', 'reviews_accepted', 'reviews_duplicate', 'reviews_quarantined', 'products_read', 'products_accepted', 'products_duplicate', 'products_quarantined', 'products_with_first_available', 'products_first_available_unparsed')}
    try:
        conn.execute('CREATE TABLE events (event_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, product_id TEXT NOT NULL, timestamp_ms INTEGER NOT NULL, rating REAL NOT NULL)')
        conn.execute('CREATE TABLE products (product_id TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL, categories TEXT NOT NULL, available_from_ms INTEGER)')
        with (stage / 'quarantine.jsonl').open('w', encoding='utf-8') as bad:
            def quarantine(source, lineno, reason):
                bad.write(json.dumps({'source': source, 'line': lineno, 'reason': reason}) + '\n')
            for line, obj, err in read_jsonl(reviews):
                counters['reviews_read'] += 1
                reason = err or valid_review(obj)
                if reason:
                    counters['reviews_quarantined'] += 1
                    quarantine('reviews', line, reason)
                    continue
                row = (fingerprint(obj), obj['user_id'], obj['parent_asin'], obj['timestamp'], obj['rating'])
                cursor = conn.execute('INSERT OR IGNORE INTO events VALUES (?,?,?,?,?)', row)
                counters['reviews_accepted' if cursor.rowcount else 'reviews_duplicate'] += 1
            for line, obj, err in read_jsonl(products):
                counters['products_read'] += 1
                reason = err or valid_product(obj)
                if reason:
                    counters['products_quarantined'] += 1
                    quarantine('products', line, reason)
                    continue
                # Allowlist of static-looking text fields. Crawl-time values
                # (average_rating, rating_number, price, bought_together) stay out.
                bits = [obj['title']]
                for name in ('description', 'features'):
                    value = obj.get(name, [])
                    if isinstance(value, list):
                        bits.extend(v for v in value if isinstance(v, str))
                # Category text is stored separately so it can be ablated.
                cats = [obj['main_category']] if nonempty_str(obj.get('main_category')) else []
                if isinstance(obj.get('categories'), list):
                    cats.extend(v for v in obj['categories'] if isinstance(v, str))
                available = first_available_ms(obj.get('details'))
                cursor = conn.execute('INSERT OR IGNORE INTO products VALUES (?,?,?,?,?)', (obj['parent_asin'], obj['title'], ' '.join(bits), ' '.join(cats), available))
                if not cursor.rowcount:
                    counters['products_duplicate'] += 1
                    continue
                counters['products_accepted'] += 1
                if available is not None:
                    counters['products_with_first_available'] += 1
                elif 'Date First Available' in str(obj.get('details', '')):
                    counters['products_first_available_unparsed'] += 1
        conn.commit()
        counters['events_with_metadata'] = conn.execute('SELECT COUNT(*) FROM events e JOIN products p ON e.product_id=p.product_id').fetchone()[0]
        counters['unique_users'] = conn.execute('SELECT COUNT(DISTINCT user_id) FROM events').fetchone()[0]
        counters['unique_interacted_products'] = conn.execute('SELECT COUNT(DISTINCT product_id) FROM events').fetchone()[0]
        bounds = conn.execute('SELECT MIN(timestamp_ms), MAX(timestamp_ms) FROM events').fetchone()
        conn.close()
        manifest = {'schema_version': 3, 'run_id': run_id, 'created_utc': datetime.now(timezone.utc).isoformat(), 'dataset': 'Amazon Reviews 2023', 'interaction_semantics': 'review, not purchase', 'timestamp_unit': 'milliseconds', 'product_metadata_temporal_status': '2023 crawl snapshot; available_from_ms is details[Date First Available], seller-reported, day precision', 'inputs': {'reviews_sha256': sha256_file(reviews), 'products_sha256': sha256_file(products)}, 'counts': counters, 'timestamp_bounds_ms': bounds, 'deduplication': 'SHA256 of user_id, parent_asin, timestamp, rating, text, asin; no true event ID'}
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        if fail_before_commit:
            raise RuntimeError('injected_failure_before_promotion')
        final = runs / run_id
        os.replace(stage, final)  # rename within same file system
        # Active pointer file changed atomically, only after all artifacts exist.
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output, delete=False) as tmp:
            tmp.write(run_id + '\n')
            temp_pointer = tmp.name
        os.replace(temp_pointer, output / 'ACTIVE')
        return manifest
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--products', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/ingestion'))
    args = parser.parse_args()
    result = ingest(args.reviews, args.products, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
