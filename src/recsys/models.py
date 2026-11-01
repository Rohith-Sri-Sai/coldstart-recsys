"""Recommendation models. All are fitted on PhaseData (pre-cutoff only).

Each model exposes scores(user) -> float32 array over data.ids. -inf means
"this model cannot score the item" and puts it after every scored item.
"""
from __future__ import annotations
import json

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from recsys.data import PhaseData


class Popularity:
    def __init__(self, data: PhaseData, cache=None):
        self.counts = data.review_counts

    def scores(self, user):
        return self.counts.copy()


class Content:
    def __init__(self, data: PhaseData, cache=None, *, use_categories=True, sublinear_tf=False,
                 min_df=1, ngram_max=1, max_features=50000):
        self.data = data
        docs = [c + (' ' + g if use_categories else '') for c, g in zip(data.content, data.categories)]
        self.text = None
        try:
            vec = TfidfVectorizer(max_features=max_features, sublinear_tf=sublinear_tf, min_df=min_df,
                                  ngram_range=(1, ngram_max), dtype=np.float32, token_pattern=r'(?u)\b\w+\b')
            self.text = vec.fit_transform(docs).tocsr()
        except ValueError:  # no usable tokens
            pass

    def scores(self, user):
        if self.text is None or user not in self.data.user_idx:
            return np.zeros(len(self.data.ids), dtype=np.float32)
        row = self.data.interactions.getrow(self.data.user_idx[user])
        # Sum of the user's item vectors; per-user scaling does not change the ranking.
        return (row @ self.text @ self.text.T).toarray().ravel().astype(np.float32)


class SVD:
    def __init__(self, data: PhaseData, cache=None, *, factors=32, seed=0):
        self.data = data
        self.unscored = data.item_users == 0
        m = data.interactions
        n = min(factors, m.shape[0] - 1, m.shape[1] - 1)
        self.svd = None
        if n >= 1 and m.nnz:
            self.svd = TruncatedSVD(n_components=n, n_iter=5, random_state=seed).fit(m)

    def scores(self, user):
        if self.svd is None or user not in self.data.user_idx:
            return np.full(len(self.data.ids), -np.inf, dtype=np.float32)
        row = self.data.interactions.getrow(self.data.user_idx[user])
        s = (self.svd.transform(row) @ self.svd.components_).ravel().astype(np.float32)
        s[self.unscored] = -np.inf
        return s


class LightGCN:
    """LightGCN (He et al., SIGIR 2020): mean of K propagation layers, BPR loss.

    Negatives are sampled uniformly from items that have interactions, without
    rejecting the user's own positives (cheap, and rare at this sparsity).
    """
    def __init__(self, data: PhaseData, cache=None, *, dim=64, layers=2, epochs=30, lr=1e-3,
                 reg=1e-4, batch_size=4096, seed=0):
        import torch  # optional dependency: pip install -e .[graph]
        self.data = data
        self.unscored = data.item_users == 0
        self.losses = []
        coo = data.interactions.tocoo()
        n_u, n_i = data.interactions.shape
        self.user_emb = np.zeros((n_u, dim), dtype=np.float32)
        self.item_emb = np.zeros((n_i, dim), dtype=np.float32)
        if coo.nnz == 0:
            return
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        # Device is picked here, not stored in params, so a config tuned on a GPU
        # still runs on a laptop.
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Symmetric-normalised bipartite adjacency: D^-1/2 A D^-1/2
        u, i = coo.row.astype(np.int64), coo.col.astype(np.int64)
        deg_u = np.bincount(u, minlength=n_u).astype(np.float32)
        deg_i = np.bincount(i, minlength=n_i).astype(np.float32)
        w = 1.0 / np.sqrt(deg_u[u] * deg_i[i])
        rows = np.concatenate([u, n_u + i]); cols = np.concatenate([n_u + i, u])
        adj = torch.sparse_coo_tensor(np.vstack([rows, cols]), np.concatenate([w, w]),
                                      (n_u + n_i, n_u + n_i), dtype=torch.float32,
                                      check_invariants=True).coalesce().to(dev)
        emb = torch.nn.Parameter(torch.randn(n_u + n_i, dim, device=dev) * 0.1)
        opt = torch.optim.Adam([emb], lr=lr)

        def propagate():
            out, layer = [emb], emb
            for _ in range(layers):
                layer = torch.sparse.mm(adj, layer)
                out.append(layer)
            return torch.stack(out).mean(0)

        neg_pool = np.flatnonzero(deg_i > 0)
        for _ in range(epochs):
            order = rng.permutation(len(u)); total = 0.0
            for start in range(0, len(order), batch_size):
                b = order[start:start + batch_size]
                bu = torch.as_tensor(u[b], device=dev)
                bp = torch.as_tensor(n_u + i[b], device=dev)
                bn = torch.as_tensor(n_u + rng.choice(neg_pool, len(b)), device=dev)
                final = propagate()
                x = (final[bu] * (final[bp] - final[bn])).sum(1)
                l2 = (emb[bu].pow(2).sum() + emb[bp].pow(2).sum() + emb[bn].pow(2).sum()) / (2 * len(b))
                loss = -torch.nn.functional.logsigmoid(x).mean() + reg * l2
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(b)
            self.losses.append(total / len(u))
        with torch.no_grad():
            final = propagate().cpu().numpy()
        self.user_emb, self.item_emb = final[:n_u], final[n_u:]

    def scores(self, user):
        if user not in self.data.user_idx:
            return np.full(len(self.data.ids), -np.inf, dtype=np.float32)
        s = self.item_emb @ self.user_emb[self.data.user_idx[user]]
        s[self.unscored] = -np.inf
        return s.astype(np.float32)


def zscore(x: np.ndarray) -> np.ndarray:
    """Per-user z-score over items the model can score; unscored items get 0."""
    out = np.zeros_like(x, dtype=np.float32)
    ok = np.isfinite(x)
    if ok.sum() > 1:
        sd = x[ok].std()
        out[ok] = (x[ok] - x[ok].mean()) / (sd if sd > 0 else 1.0)
    return out


class _Hybrid:
    def __init__(self, data: PhaseData, cache=None, *, content=None, lightgcn=None):
        self.data = data
        self.content = build('content', data, content or {}, cache)
        self.graph = build('lightgcn', data, lightgcn or {}, cache)

    def parts(self, user):
        return zscore(self.content.scores(user)), zscore(self.graph.scores(user))


class FixedHybrid(_Hybrid):
    def __init__(self, data, cache=None, *, alpha=0.5, **parts):
        super().__init__(data, cache, **parts)
        self.alpha = alpha

    def scores(self, user):
        c, g = self.parts(user)
        return self.alpha * c + (1 - self.alpha) * g


class AdaptiveHybrid(_Hybrid):
    """Graph weight grows with item support: w = n / (n + beta)."""
    def __init__(self, data, cache=None, *, beta=5.0, **parts):
        super().__init__(data, cache, **parts)
        self.w = data.item_users / (data.item_users + beta)

    def scores(self, user):
        c, g = self.parts(user)
        return ((1 - self.w) * c + self.w * g).astype(np.float32)


MODELS = {'popularity': Popularity, 'content': Content, 'svd': SVD, 'lightgcn': LightGCN,
          'fixed_hybrid': FixedHybrid, 'adaptive_hybrid': AdaptiveHybrid}


def build(name: str, data: PhaseData, params: dict, cache: dict | None = None):
    """Fit a model, reusing an already fitted one with identical params."""
    key = (name, json.dumps(params, sort_keys=True))
    if cache is not None and key in cache:
        return cache[key]
    model = MODELS[name](data, cache, **params)
    if cache is not None:
        cache[key] = model
    return model


def top_k(scores: np.ndarray, exclude: np.ndarray, tiebreak: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k best non-excluded items, ties broken by `tiebreak`. O(n) plus a small sort."""
    mask = np.ones(len(scores), dtype=bool); mask[exclude] = False
    cand = np.flatnonzero(mask)
    if len(cand) == 0:
        return cand
    s = scores[cand]
    m = min(k, len(cand))
    kth = np.partition(-s, m - 1)[m - 1]
    sel = np.flatnonzero(-s <= kth)          # every item tied with the k-th is kept
    order = np.lexsort((tiebreak[cand[sel]], -s[sel]))
    return cand[sel[order]][:k]
