"""Equal-budget tuning on the dev window, following docs/PROTOCOL.md.

Never reads the test window. Writes runs/<id>/ with trials.jsonl (one line per
fitted config), selected.json (pass to recsys-evaluate --config for the final
test run), dev_report.json and summary.md, then moves the ACTIVE pointer.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import uuid

from recsys.data import load_phase
from recsys.evaluate import evaluate_data
from recsys.split import sha256_file
from recsys.tracking import JsonlTracker

# Simplest first: ties go to the earlier entry. At most 9 configs per model.
GRIDS = {
    'popularity': [{}],
    'content': [{}, {'sublinear_tf': True}, {'min_df': 2}, {'sublinear_tf': True, 'min_df': 2},
                {'ngram_max': 2}, {'ngram_max': 2, 'sublinear_tf': True}, {'ngram_max': 2, 'min_df': 2},
                {'ngram_max': 2, 'sublinear_tf': True, 'min_df': 2}],
    'svd': [{'factors': f} for f in (8, 16, 32, 64, 128, 256)],
    'lightgcn': [{'layers': n, 'reg': r} for n in (1, 2, 3) for r in (1e-4, 1e-3)],
    'fixed_hybrid': [{'alpha': a} for a in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)],
    'adaptive_hybrid': [{'beta': b} for b in (0.5, 1, 2, 3, 5, 10, 20, 50, 100)],
}
COMPARE = [['adaptive_hybrid', 'fixed_hybrid'], ['adaptive_hybrid', 'lightgcn'], ['adaptive_hybrid', 'content']]


def peak_rss_mb():
    try:
        import resource  # not available on Windows
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == 'darwin' else 1024), 1)


def tune(split_root: Path, output_root: Path, *, epochs: int = 30, seeds=(0, 1, 2), k: int = 10,
         n_boot: int = 1000, tracker=None) -> dict:
    split_root, output_root = Path(split_root), Path(output_root)
    split_id = (split_root/'ACTIVE').read_text(encoding='utf8').strip()
    db_path = split_root/'runs'/split_id/'split.sqlite'
    data = load_phase(db_path, 'dev')
    output_root.mkdir(parents=True, exist_ok=True)
    runs = output_root/'runs'; runs.mkdir(exist_ok=True)
    run_id = str(uuid.uuid4()); stage = runs/('.building-'+run_id); stage.mkdir()
    trackers = [JsonlTracker(stage/'trials.jsonl')] + ([tracker] if tracker else [])
    metric = f'dev_ndcg_at_{k}'
    cache, started = {}, time.perf_counter()

    def trial(stage_tag, label, model, params):
        t0 = time.perf_counter()
        ev = evaluate_data(data, {'models': {label: {'model': model, 'params': params}}}, k=k, n_boot=0, cache=cache)
        cohorts = ev.report['models'][label]['cohorts']
        metrics = {f'dev_{m}_{c}': v for c, row in cohorts.items()
                   for m, v in (('ndcg', row['ndcg_at_k']), ('recall', row['recall_at_k'])) if v is not None}
        metrics[metric] = cohorts['all']['ndcg_at_k'] or 0.0
        metrics['seconds'] = round(time.perf_counter() - t0, 3)
        for t in trackers:
            t.log(f'{stage_tag}/{label}', params, metrics, {'stage': stage_tag, 'model': label, 'split_run_id': split_id})
        return metrics[metric]

    def search(model, extra):
        best, best_score = None, float('-inf')
        for entry in GRIDS[model]:
            params = {**entry, **extra}
            score = trial('grid', model, model, params)
            if score > best_score:              # strict: earlier entry wins ties
                best, best_score = params, score
        # Keep only the chosen fit of this model (hybrids need the chosen components).
        for key in [key for key in cache if key[0] == model and key[1] != json.dumps(best, sort_keys=True)]:
            del cache[key]
        return best

    selected = {}
    for model in ('popularity', 'content', 'svd'):
        selected[model] = search(model, {})
    selected['lightgcn'] = search('lightgcn', {'epochs': epochs, 'seed': 0})
    parts = {'content': selected['content'], 'lightgcn': selected['lightgcn']}
    for model in ('fixed_hybrid', 'adaptive_hybrid'):
        selected[model] = search(model, parts)

    seed_summary = {}
    for model in ('lightgcn', 'fixed_hybrid', 'adaptive_hybrid'):
        scores = []
        for s in seeds:
            graph = {**selected['lightgcn'], 'seed': s}
            params = graph if model == 'lightgcn' else {**selected[model], 'lightgcn': graph}
            scores.append(trial('seed', model, model, params))
        seed_summary[model] = {metric: scores, 'mean': statistics.mean(scores),
                               'sd': statistics.stdev(scores) if len(scores) > 1 else 0.0}
    ablation = {'content_no_categories': trial('ablation', 'content_no_categories', 'content',
                                               {**selected['content'], 'use_categories': False}),
                'content_with_categories': trial('ablation', 'content_with_categories', 'content', selected['content'])}

    config = {'models': {m: {'model': m, 'params': p} for m, p in selected.items()}, 'compare': COMPARE,
              'selected_on': {'phase': 'dev', 'split_run_id': split_id, 'rule': f'max {metric}, earliest grid entry on ties'}}
    final = evaluate_data(data, config, k=k, n_boot=n_boot, cache=cache).report
    (stage/'selected.json').write_text(json.dumps(config, indent=2), encoding='utf8')
    (stage/'dev_report.json').write_text(json.dumps(final, indent=2), encoding='utf8')
    (stage/'summary.md').write_text(summary_markdown(final, seed_summary, ablation, metric), encoding='utf8')
    result = {'schema_version': 1, 'tuning_run_id': run_id, 'phase': 'dev', 'split_run_id': split_id,
              'split_sha256': sha256_file(db_path), 'k': k, 'epochs': epochs, 'seeds': list(seeds),
              'selected': selected, 'seed_summary': seed_summary, 'ablation': ablation,
              'total_seconds': round(time.perf_counter() - started, 1), 'peak_rss_mb': peak_rss_mb(),
              'evaluated_users': final['evaluated_users'], 'catalog_size': len(data.ids)}
    (stage/'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    os.replace(stage, runs/run_id)
    with tempfile.NamedTemporaryFile('w', dir=output_root, delete=False, encoding='utf8') as f:
        f.write(run_id+'\n'); pointer = f.name
    os.replace(pointer, output_root/'ACTIVE')
    return result


def _fmt(value, ci=None):
    if value is None:
        return 'n/a'
    return f'{value:.4f}' + (f' [{ci[0]:.4f}, {ci[1]:.4f}]' if ci else '')


def summary_markdown(report, seed_summary, ablation, metric) -> str:
    lines = ['# Dev results', '', f"K={report['k']}, {report['evaluated_users']} users, 95% bootstrap intervals.", '',
             '| model | NDCG all | NDCG cold | NDCG sparse | NDCG warm | coverage | cold slot share |',
             '|---|---|---|---|---|---|---|']
    for name, row in report['models'].items():
        c = row['cohorts']
        cells = [_fmt(c[x]['ndcg_at_k'], c[x]['ndcg_ci95']) for x in ('all', 'cold', 'sparse', 'warm')]
        lines.append(f"| {name} | {' | '.join(cells)} | {_fmt(row['catalog_coverage_at_k'])} | {_fmt(row['cold_slot_share'])} |")
    lines += ['', '## Paired differences (NDCG)', '', '| comparison | all | sparse | warm |', '|---|---|---|---|']
    for name, row in report['comparisons'].items():
        lines.append(f"| {name} | " + ' | '.join(_fmt(row[c]['ndcg_diff'], row[c]['ndcg_diff_ci95']) for c in ('all', 'sparse', 'warm')) + ' |')
    lines += ['', f'## Seed repeats ({metric})', '', '| model | mean | sd | runs |', '|---|---|---|---|']
    for name, row in seed_summary.items():
        lines.append(f"| {name} | {row['mean']:.4f} | {row['sd']:.4f} | {len(row[metric])} |")
    lines += ['', '## Category ablation (content, NDCG all)', '']
    lines += [f'- {name}: {value:.4f}' for name, value in ablation.items()]
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--splits', type=Path, default=Path('artifacts/splits'))
    p.add_argument('--output', type=Path, default=Path('artifacts/tuning'))
    p.add_argument('--epochs', type=int, default=30, help='LightGCN epochs, same for every config')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--bootstrap', type=int, default=1000)
    a = p.parse_args()
    out = tune(a.splits, a.output, epochs=a.epochs, seeds=a.seeds, k=a.k, n_boot=a.bootstrap)
    print(json.dumps({key: out[key] for key in ('tuning_run_id', 'selected', 'total_seconds', 'peak_rss_mb')}, indent=2))


if __name__ == '__main__':
    main()
