from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from recsys.data import PhaseData


def make_data(products: dict, history: dict, labels=None, categories=None) -> PhaseData:
    """Tiny in-memory PhaseData for unit tests. products: id -> text."""
    ids = sorted(products); index = {p: i for i, p in enumerate(ids)}
    users = sorted(history); user_idx = {u: i for i, u in enumerate(users)}
    rows = [(user_idx[u], index[p]) for u in users for p in history[u] if p in index]
    m = csr_matrix((np.ones(len(rows), dtype=np.float32), ([r for r, _ in rows], [c for _, c in rows])),
                   shape=(len(users), len(ids)), dtype=np.float32)
    item_users = np.asarray(m.sum(axis=0)).ravel().astype(np.float32)
    counts = Counter(p for u in users for p in history[u])
    cohort = {p: 'cold' if n == 0 else 'sparse' if n <= 5 else 'warm' for p, n in zip(ids, item_users)}
    return PhaseData(phase='dev', cutoff_ms=0, ids=ids, index=index, users=users, user_idx=user_idx,
                     history=history, interactions=m,
                     review_counts=np.array([counts[p] for p in ids], dtype=np.float32),
                     item_users=item_users, content=[products[p] for p in ids],
                     categories=[(categories or {}).get(p, '') for p in ids],
                     cohort=cohort, labels=labels or {})
