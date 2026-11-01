import unittest
import numpy as np
from recsys.models import (AdaptiveHybrid, Content, FixedHybrid, LightGCN, Popularity, SVD,
                           build, top_k, zscore)
from helpers import make_data


def rank(model, data, user, k, seed=0):
    tiebreak = np.random.default_rng(seed).permutation(len(data.ids))
    return [data.ids[i] for i in top_k(model.scores(user), data.seen_idx(user), tiebreak, k)]


def group_history(n):
    """Users 0-9 like a*, users 10-19 like b*. Odd users have not seen the group's first item yet."""
    items = ['a1', 'a2', 'a3'] if n < 10 else ['b1', 'b2', 'b3']
    return set(items[1:]) if n % 2 else set(items)


GROUPS = {f'u{n}': group_history(n) for n in range(20)}  # 'new' below has no reviews
GROUP_ITEMS = {p: f'{p[0]} item' for p in ('a1', 'a2', 'a3', 'b1', 'b2', 'b3')} | {'new': 'a item'}


class TopKTests(unittest.TestCase):
    def test_excludes_seen_and_keeps_ties_at_the_boundary(self):
        scores = np.array([5, 1, 1, 1, 9], dtype=np.float32)
        got = top_k(scores, np.array([4]), np.array([0, 3, 2, 1, 4]), 3)
        self.assertEqual(got.tolist(), [0, 3, 2])  # item 4 seen; ties 1/2/3 ordered by tiebreak

    def test_ties_are_seeded_not_alphabetical(self):
        data = make_data({chr(97+i): 'x' for i in range(26)}, {'u': set()})
        model = Popularity(data)
        self.assertEqual(rank(model, data, 'u', 26), rank(model, data, 'u', 26))
        self.assertNotEqual(rank(model, data, 'u', 26), sorted(data.ids))


class BaselineTests(unittest.TestCase):
    def test_content_can_retrieve_unseen_item(self):
        data = make_data({'a': 'red soap', 'b': 'red soap moisturizing', 'c': 'blue keyboard'},
                         {'u1': {'a'}, 'u2': {'c'}})
        self.assertEqual(rank(Content(data), data, 'u1', 1), ['b'])

    def test_category_ablation_changes_the_text(self):
        data = make_data({'a': 'soap', 'b': 'lamp', 'c': 'pen'}, {'u1': {'a'}},
                         categories={'a': 'bath', 'b': 'bath', 'c': 'office'})
        self.assertEqual(rank(Content(data, use_categories=True), data, 'u1', 1), ['b'])
        self.assertEqual(Content(data, use_categories=False).scores('u1')[data.index['b']], 0.0)

    def test_svd_and_lightgcn_rank_unscored_items_last(self):
        data = make_data(GROUP_ITEMS, GROUPS)
        for model in (SVD(data, factors=2), LightGCN(data, epochs=5, batch_size=16)):
            self.assertEqual(rank(model, data, 'u1', 10)[-1], 'new')


class LightGCNTests(unittest.TestCase):
    def test_learns_group_structure(self):
        data = make_data(GROUP_ITEMS, GROUPS)
        model = LightGCN(data, dim=16, layers=2, epochs=150, lr=0.05, batch_size=64, seed=0)
        self.assertLess(model.losses[-1], model.losses[0])
        self.assertEqual(rank(model, data, 'u1', 1), ['a1'])   # u1 is in the a-group and has not seen a1
        self.assertEqual(rank(model, data, 'u11', 1), ['b1'])

    def test_same_seed_same_embeddings(self):
        data = make_data(GROUP_ITEMS, GROUPS)
        a, b = (LightGCN(data, epochs=3, batch_size=16, seed=7) for _ in range(2))
        np.testing.assert_allclose(a.item_emb, b.item_emb)


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.data = make_data(GROUP_ITEMS, GROUPS)
        self.cache = {}
        self.parts = {'content': {}, 'lightgcn': {'epochs': 3, 'batch_size': 16}}

    def test_zscore_ignores_unscored_items(self):
        z = zscore(np.array([1.0, 3.0, -np.inf], dtype=np.float32))
        np.testing.assert_allclose(z, [-1.0, 1.0, 0.0])

    def test_fixed_alpha_one_is_content(self):
        hybrid = FixedHybrid(self.data, self.cache, alpha=1.0, **self.parts)
        content = build('content', self.data, {}, self.cache)
        self.assertEqual(rank(hybrid, self.data, 'u1', 7), rank(content, self.data, 'u1', 7))

    def test_adaptive_uses_only_content_for_cold_items(self):
        hybrid = AdaptiveHybrid(self.data, self.cache, beta=5.0, **self.parts)
        c, _ = hybrid.parts('u1')
        new = self.data.index['new']
        self.assertEqual(hybrid.w[new], 0.0)
        self.assertAlmostEqual(float(hybrid.scores('u1')[new]), float(c[new]), places=5)

    def test_components_are_fitted_once(self):
        FixedHybrid(self.data, self.cache, alpha=0.3, **self.parts)
        AdaptiveHybrid(self.data, self.cache, beta=2.0, **self.parts)
        self.assertEqual(sorted(name for name, _ in self.cache), ['content', 'lightgcn'])


if __name__ == '__main__':
    unittest.main()
