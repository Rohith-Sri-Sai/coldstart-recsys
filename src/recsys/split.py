"""Chronological split with a pre-cutoff catalog.

train: t < train_end; dev: train_end <= t < dev_end; test: dev_end <= t < test_end.
Events at or after test_end are ignored.

For each evaluation window, everything is decided at the window START (cutoff):
- target users: users with at least one event before the cutoff;
- cohorts: distinct users who reviewed the product before the cutoff
  (0 cold, 1-5 sparse, 6+ warm);
- catalog eligibility: Date First Available <= cutoff, OR reviewed before the cutoff.
  Products with neither are excluded. Reviews inside the window never make a
  product eligible, because that would reveal the label.

Date First Available comes from the 2023 metadata crawl and is seller-reported,
so eligibility is a proxy for listing time, not ground truth.
"""
from __future__ import annotations
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import uuid


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def build_split(ingestion_root: Path, output_root: Path, train_end_ms: int, dev_end_ms: int,
                test_end_ms: int, *, fail_before_commit=False):
    ingestion_root, output_root = Path(ingestion_root), Path(output_root)
    cutoffs = (train_end_ms, dev_end_ms, test_end_ms)
    if any(type(c) is not int for c in cutoffs) or not 0 < train_end_ms < dev_end_ms < test_end_ms:
        raise ValueError('cutoffs must be increasing positive integer milliseconds')
    source_id = (ingestion_root / 'ACTIVE').read_text(encoding='utf-8').strip()
    if not source_id or '/' in source_id or '\\' in source_id or source_id.startswith('.'):
        raise ValueError('invalid ingestion run pointer')
    source = ingestion_root / 'runs' / source_id / 'records.sqlite'
    if not source.is_file():
        raise FileNotFoundError(source)
    output_root.mkdir(parents=True, exist_ok=True)
    runs = output_root / 'runs'
    runs.mkdir(exist_ok=True)
    run_id = str(uuid.uuid4())
    stage = runs / ('.building-' + run_id)
    stage.mkdir()
    db_file = stage / 'split.sqlite'
    windows = [('train', None, train_end_ms), ('dev', train_end_ms, dev_end_ms), ('test', dev_end_ms, test_end_ms)]
    # closing(): sqlite3's own context manager commits but does not close, and an
    # open handle blocks os.replace on Windows.
    with closing(sqlite3.connect(f'file:{source}?mode=ro', uri=True)) as src, closing(sqlite3.connect(db_file)) as db:
        src.backup(db)
        db.executescript('''
        CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp_ms);
        CREATE INDEX IF NOT EXISTS idx_events_product_time ON events(product_id,timestamp_ms);
        CREATE INDEX IF NOT EXISTS idx_events_user_time ON events(user_id,timestamp_ms);
        CREATE TABLE split_events (event_id TEXT PRIMARY KEY, phase TEXT NOT NULL,
            FOREIGN KEY(event_id) REFERENCES events(event_id));
        CREATE TABLE phase_cohorts (phase TEXT NOT NULL, product_id TEXT NOT NULL,
            prior_users INTEGER NOT NULL, cohort TEXT NOT NULL, eligible INTEGER NOT NULL,
            PRIMARY KEY(phase, product_id));
        CREATE TABLE phase_cutoffs (phase TEXT PRIMARY KEY, start_ms INTEGER, end_ms INTEGER NOT NULL);
        ''')
        for phase, start, end in windows:
            db.execute('INSERT INTO phase_cutoffs VALUES (?,?,?)', (phase, start, end))
            if phase == 'train':
                db.execute('INSERT INTO split_events SELECT event_id, ? FROM events WHERE timestamp_ms < ?', (phase, end))
                continue
            db.execute('''INSERT INTO split_events SELECT event_id, ? FROM events
                WHERE timestamp_ms >= ? AND timestamp_ms < ?
                AND EXISTS (SELECT 1 FROM events prior WHERE prior.user_id=events.user_id AND prior.timestamp_ms < ?)''',
                (phase, start, end, start))
            db.execute('''INSERT INTO phase_cohorts
                SELECT ?, p.product_id, COUNT(DISTINCT e.user_id),
                CASE WHEN COUNT(DISTINCT e.user_id)=0 THEN 'cold'
                     WHEN COUNT(DISTINCT e.user_id)<=5 THEN 'sparse' ELSE 'warm' END,
                CASE WHEN p.available_from_ms <= ? OR COUNT(e.event_id) > 0 THEN 1 ELSE 0 END
                FROM products p LEFT JOIN events e ON e.product_id=p.product_id AND e.timestamp_ms < ?
                GROUP BY p.product_id''', (phase, start, start))
        stats = {phase: db.execute('SELECT COUNT(*) FROM split_events WHERE phase=?', (phase,)).fetchone()[0]
                 for phase, _, _ in windows}
        stats['ignored_after_test_end'] = db.execute('SELECT COUNT(*) FROM events WHERE timestamp_ms >= ?', (test_end_ms,)).fetchone()[0]
        stats['products'] = db.execute('SELECT COUNT(*) FROM products').fetchone()[0]
        stats['products_with_first_available'] = db.execute('SELECT COUNT(*) FROM products WHERE available_from_ms IS NOT NULL').fetchone()[0]
        stats['first_available_after_first_review'] = db.execute('''SELECT COUNT(*) FROM products p
            JOIN (SELECT product_id, MIN(timestamp_ms) first_ms FROM events GROUP BY product_id) f USING(product_id)
            WHERE p.available_from_ms > f.first_ms''').fetchone()[0]
        for phase in ('dev', 'test'):
            labels = '''FROM split_events s JOIN events e USING(event_id)
                LEFT JOIN phase_cohorts c ON c.product_id=e.product_id AND c.phase=s.phase WHERE s.phase=?'''
            stats[phase+'_labels_no_metadata'] = db.execute('SELECT COUNT(*) '+labels+' AND c.product_id IS NULL', (phase,)).fetchone()[0]
            stats[phase+'_labels_dropped_ineligible'] = db.execute('SELECT COUNT(*) '+labels+' AND c.eligible=0', (phase,)).fetchone()[0]
            stats[phase+'_label_cohorts'] = dict(db.execute('SELECT c.cohort, COUNT(*) '+labels+' AND c.eligible=1 GROUP BY c.cohort', (phase,)).fetchall())
            stats[phase+'_catalog_cohorts'] = dict(db.execute('SELECT cohort, COUNT(*) FROM phase_cohorts WHERE phase=? AND eligible=1 GROUP BY cohort', (phase,)).fetchall())
        db.commit()
    manifest = {'schema_version': 2, 'split_run_id': run_id, 'ingestion_run_id': source_id,
        'source_db_sha256': sha256_file(source),
        'cutoffs_ms': {'train_end_exclusive': train_end_ms, 'dev_end_exclusive': dev_end_ms, 'test_end_exclusive': test_end_ms},
        'intervals': 'train < train_end; dev [train_end, dev_end); test [dev_end, test_end)',
        'evaluated_users': 'users with any event strictly before the window start',
        'cohort_definition': 'distinct users reviewing the product strictly before window start: 0 cold, 1-5 sparse, 6+ warm',
        'catalog_rule': 'eligible at window start if Date First Available <= start or reviewed before start',
        'label_semantics': 'observed review interactions, not purchases',
        'known_limitations': ['Date First Available is seller-reported and from a 2023 crawl',
                              'product text is a 2023 snapshot and may differ from the text at the cutoff',
                              'no stock or delisting dates: once eligible, always eligible'],
        'stats': stats}
    (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if fail_before_commit:
        raise RuntimeError('injected_failure_before_split_promotion')
    os.replace(stage, runs / run_id)
    with tempfile.NamedTemporaryFile('w', dir=output_root, delete=False, encoding='utf-8') as tmp:
        tmp.write(run_id + '\n')
        pointer = tmp.name
    os.replace(pointer, output_root / 'ACTIVE')
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ingestion', type=Path, default=Path('artifacts/ingestion'))
    p.add_argument('--output', type=Path, default=Path('artifacts/splits'))
    p.add_argument('--train-end-ms', type=int, required=True)
    p.add_argument('--dev-end-ms', type=int, required=True)
    p.add_argument('--test-end-ms', type=int, required=True)
    a = p.parse_args()
    print(json.dumps(build_split(a.ingestion, a.output, a.train_end_ms, a.dev_end_ms, a.test_end_ms), indent=2))


if __name__ == '__main__':
    main()
