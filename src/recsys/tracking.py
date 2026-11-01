"""Minimal experiment tracking: one JSON line per run.

To switch to MLflow later, write a class with the same log() signature, e.g.

    with mlflow.start_run(run_name=name):
        mlflow.log_params(flatten(params)); mlflow.log_metrics(metrics)
        mlflow.set_tags(tags or {})

and pass it to tune() instead of JsonlTracker.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path


def flatten(params: dict, prefix: str = '') -> dict:
    """{'lightgcn': {'layers': 2}} -> {'lightgcn.layers': 2} (MLflow params are flat)."""
    out = {}
    for key, value in params.items():
        name = f'{prefix}{key}'
        if isinstance(value, dict):
            out.update(flatten(value, name + '.'))
        else:
            out[name] = value
    return out


class JsonlTracker:
    def __init__(self, path: Path):
        self.path = Path(path)

    def log(self, name: str, params: dict, metrics: dict, tags: dict | None = None):
        row = {'utc': datetime.now(timezone.utc).isoformat(), 'name': name,
               'params': params, 'metrics': metrics, 'tags': tags or {}}
        with self.path.open('a', encoding='utf8') as f:
            f.write(json.dumps(row) + '\n')
