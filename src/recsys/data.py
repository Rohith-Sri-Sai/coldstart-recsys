"""Load one evaluation window from a split database.

Everything here uses only events before the window's cutoff, except `labels`,
which are the held-out reviews inside the window.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class PhaseData:
    phase: str
    cutoff_ms: int
    ids: list[str]                      # eligible catalog, sorted
    index: dict[str, int]
    users: list[str]                    # users with history before the cutoff
    user_idx: dict[str, int]
    history: dict[str, set[str]]        # all products reviewed before the cutoff
    interactions: csr_matrix            # binary users x eligible items
    review_counts: np.ndarray           # review events per item before the cutoff
    item_users: np.ndarray              # distinct users per item before the cutoff
    content: list[str]
    categories: list[str]
    cohort: dict[str, str]              # item -> cold / sparse / warm
    labels: dict[str, dict[str, set[str]]]  # user -> cohort -> held-out items

    def seen_idx(self, user: str) -> np.ndarray:
        return np.array([self.index[p] for p in self.history.get(user, ()) if p in self.index], dtype=np.int64)


def load_phase(split_db: Path, phase: str) -> PhaseData:
    if phase not in ('dev', 'test'):
        raise ValueError('phase must be dev or test')
    with closing(sqlite3.connect(f'file:{Path(split_db)}?mode=ro', uri=True)) as db:
        cutoff = db.execute('SELECT start_ms FROM phase_cutoffs WHERE phase=?', (phase,)).fetchone()[0]
        rows = db.execute('''SELECT p.product_id, p.content, p.categories, c.cohort FROM products p
            JOIN phase_cohorts c ON c.product_id=p.product_id AND c.phase=?
            WHERE c.eligible=1 ORDER BY p.product_id''', (phase,)).fetchall()
        history = defaultdict(set); counts = Counter()
        for user, product in db.execute('SELECT user_id, product_id FROM events WHERE timestamp_ms < ?', (cutoff,)):
            history[user].add(product)
            counts[product] += 1
        labels = defaultdict(lambda: defaultdict(set))
        for user, product, cohort in db.execute('''SELECT e.user_id, e.product_id, c.cohort FROM split_events s
                JOIN events e ON s.event_id=e.event_id
                JOIN phase_cohorts c ON c.product_id=e.product_id AND c.phase=s.phase
                WHERE s.phase=? AND c.eligible=1''', (phase,)):
            if user in history and product not in history[user]:
                labels[user][cohort].add(product)
    ids = [r[0] for r in rows]
    index = {p: i for i, p in enumerate(ids)}
    users = sorted(history)
    user_idx = {u: i for i, u in enumerate(users)}
    r_idx, c_idx = [], []
    for u in users:
        for p in history[u]:
            if p in index:
                r_idx.append(user_idx[u]); c_idx.append(index[p])
    interactions = csr_matrix((np.ones(len(r_idx), dtype=np.float32), (r_idx, c_idx)),
                              shape=(len(users), len(ids)), dtype=np.float32)
    return PhaseData(
        phase=phase, cutoff_ms=cutoff, ids=ids, index=index, users=users, user_idx=user_idx,
        history=dict(history), interactions=interactions,
        review_counts=np.array([counts[p] for p in ids], dtype=np.float32),
        item_users=np.asarray(interactions.sum(axis=0)).ravel().astype(np.float32),
        content=[r[1] for r in rows], categories=[r[2] for r in rows],
        cohort={r[0]: r[3] for r in rows},
        labels={u: dict(c) for u, c in sorted(labels.items())})
