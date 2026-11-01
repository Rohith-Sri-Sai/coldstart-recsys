"""Chronological evaluation.

For a window starting at cutoff T (dev: train_end, test: dev_end):
- models are fitted on every event before T (dev: train; test: train + dev),
  with the settings chosen on dev reused unchanged for test;
- candidates are products eligible at T minus what the user reviewed before T;
- labels are eligible, unseen products the user reviews inside the window.
See docs/PROTOCOL.md for tuning and the hypothesis check.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np

from recsys.data import PhaseData, load_phase
from recsys.models import build, top_k
from recsys.split import sha256_file

COHORTS = ('all', 'cold', 'sparse', 'warm')
DEFAULT_CONFIG = {'models': {'popularity': {'model': 'popularity', 'params': {}},
                             'content': {'model': 'content', 'params': {}},
                             'svd': {'model': 'svd', 'params': {}}},
                  'compare': []}


def metrics(ranking: list[str], truth: set[str], k: int) -> tuple[float, float]:
    """Recall@k (hits / |truth|) and NDCG@k with binary relevance."""
    if not truth:
        raise ValueError('truth must be nonempty')
    hits = sum(p in truth for p in ranking[:k])
    dcg = sum(1.0/math.log2(i+2) for i, p in enumerate(ranking[:k]) if p in truth)
    idcg = sum(1.0/math.log2(i+2) for i in range(min(k, len(truth))))
    return hits/len(truth), dcg/idcg


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> list[float] | None:
    if len(values) == 0 or n_boot <= 0:
        return None
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


@dataclass(frozen=True)
class Evaluation:
    report: dict
    rankings: dict


def evaluate_data(data: PhaseData, config: dict, *, k: int = 10, seed: int = 0,
                  n_boot: int = 1000, cache: dict | None = None) -> Evaluation:
    if type(k) is not int or k < 1:
        raise ValueError('k must be a positive integer')
    tiebreak = np.random.default_rng(seed).permutation(len(data.ids))
    cold_items = np.array([data.cohort[p] == 'cold' for p in data.ids], dtype=bool)
    cache = {} if cache is None else cache
    results, rankings, per_user = {}, {}, {}
    for label, spec in config['models'].items():
        model = build(spec['model'], data, spec.get('params', {}), cache)
        values = {c: {'recall': [], 'ndcg': []} for c in COHORTS}
        ranking_by_user, exposure, cold_slots, slots = {}, set(), 0, 0
        for user, cohort_labels in data.labels.items():
            top = top_k(model.scores(user), data.seen_idx(user), tiebreak, k)
            ranked = [data.ids[i] for i in top]
            ranking_by_user[user] = ranked
            exposure.update(top.tolist()); cold_slots += int(cold_items[top].sum()); slots += len(top)
            truth_all = set().union(*cohort_labels.values())
            for cohort in COHORTS:
                truth = truth_all if cohort == 'all' else cohort_labels.get(cohort)
                if truth:
                    r, n = metrics(ranked, truth, k)
                    values[cohort]['recall'].append(r); values[cohort]['ndcg'].append(n)
        rng = np.random.default_rng(seed)
        by_cohort = {}
        for cohort in COHORTS:
            rec, ndcg = np.array(values[cohort]['recall']), np.array(values[cohort]['ndcg'])
            by_cohort[cohort] = {'users': len(rec),
                                 'recall_at_k': float(rec.mean()) if len(rec) else None,
                                 'ndcg_at_k': float(ndcg.mean()) if len(ndcg) else None,
                                 'recall_ci95': bootstrap_ci(rec, rng, n_boot),
                                 'ndcg_ci95': bootstrap_ci(ndcg, rng, n_boot)}
        results[label] = {'model': spec['model'], 'params': spec.get('params', {}), 'cohorts': by_cohort,
                          'catalog_coverage_at_k': len(exposure)/len(data.ids) if data.ids else None,
                          'cold_slot_share': cold_slots/slots if slots else None,
                          'catalog_size': len(data.ids)}
        rankings[label] = ranking_by_user
        per_user[label] = values
    comparisons = {}
    for a, b in config.get('compare', []):
        rng = np.random.default_rng(seed)
        comparisons[f'{a} - {b}'] = {
            c: {'ndcg_diff': float(d.mean()) if len(d) else None, 'ndcg_diff_ci95': bootstrap_ci(d, rng, n_boot)}
            for c in COHORTS
            for d in [np.array(per_user[a][c]['ndcg']) - np.array(per_user[b][c]['ndcg'])]}
    report = {'schema_version': 3, 'phase': data.phase, 'k': k, 'seed': seed, 'bootstrap_resamples': n_boot,
              'fit_events_before_ms': data.cutoff_ms,
              'evaluated_users': len(data.labels),
              'candidate_rule': 'products eligible at the cutoff, minus items the user reviewed before it',
              'label_rule': 'eligible, unseen products reviewed in the window by users with history before the cutoff',
              'denominator': 'macro average over users with >=1 label in the cohort; null means no such users',
              'label_semantics': 'reviews, not purchases; low-star reviews still count',
              'models': results, 'comparisons': comparisons}
    return Evaluation(report, rankings)


def evaluate(split_db: Path, *, k: int = 10, phase: str = 'dev', config: dict | None = None,
             seed: int = 0, n_boot: int = 1000) -> Evaluation:
    return evaluate_data(load_phase(split_db, phase), config or DEFAULT_CONFIG, k=k, seed=seed, n_boot=n_boot)


def evaluate_release(split_root: Path, output_root: Path, *, k: int = 10, phase: str = 'dev',
                     config: dict | None = None, seed: int = 0, n_boot: int = 1000,
                     fail_before_commit=False) -> dict:
    split_root = Path(split_root); output_root = Path(output_root)
    config = config or DEFAULT_CONFIG
    run_id = (split_root/'ACTIVE').read_text(encoding='utf8').strip()
    if not run_id or '/' in run_id or '\\' in run_id or run_id.startswith('.'):
        raise ValueError('invalid split pointer')
    db_path = split_root/'runs'/run_id/'split.sqlite'
    if not db_path.is_file(): raise FileNotFoundError(db_path)
    r = evaluate(db_path, k=k, phase=phase, config=config, seed=seed, n_boot=n_boot)
    output_root.mkdir(parents=True, exist_ok=True)
    runs = output_root/'runs'; runs.mkdir(exist_ok=True)
    new_id = str(uuid.uuid4()); stage = runs/('.building-'+new_id); stage.mkdir()
    split_sha = sha256_file(db_path)
    result = {**r.report, 'config': config, 'split_run_id': run_id, 'split_sha256': split_sha, 'evaluation_run_id': new_id}
    (stage/'report.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    (stage/'rankings.json').write_text(json.dumps(r.rankings), encoding='utf8')
    if fail_before_commit: raise RuntimeError('injected_failure_before_evaluation_promotion')
    os.replace(stage, runs/new_id)
    with tempfile.NamedTemporaryFile('w', dir=output_root, delete=False, encoding='utf8') as f:
        f.write(new_id+'\n'); pointer = f.name
    os.replace(pointer, output_root/'ACTIVE')
    if phase == 'test':
        # Test should be run once with the config chosen on dev. Logging every run
        # makes repeated peeking visible instead of silent.
        frozen = json.dumps({'k': k, 'seed': seed, 'config': config, 'split_sha256': split_sha}, sort_keys=True)
        entry = {'utc': datetime.now(timezone.utc).isoformat(), 'evaluation_run_id': new_id,
                 'config_sha256': hashlib.sha256(frozen.encode()).hexdigest()}
        with (output_root/'test_runs.jsonl').open('a', encoding='utf8') as log:
            log.write(json.dumps(entry)+'\n')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--splits', type=Path, default=Path('artifacts/splits'))
    p.add_argument('--output', type=Path, default=Path('artifacts/evaluation'))
    p.add_argument('--phase', choices=['dev', 'test'], default='dev')
    p.add_argument('--config', type=Path, help='JSON with "models" and optional "compare" (default: 3 baselines)')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--bootstrap', type=int, default=1000)
    a = p.parse_args()
    config = json.loads(a.config.read_text(encoding='utf8')) if a.config else None
    out = evaluate_release(a.splits, a.output, k=a.k, phase=a.phase, config=config, seed=a.seed, n_boot=a.bootstrap)
    print(json.dumps({m: v['cohorts'] for m, v in out['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
