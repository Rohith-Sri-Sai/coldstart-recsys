import json
from pathlib import Path
import tempfile
import unittest
from recsys.tracking import JsonlTracker, flatten
from recsys.tune import GRIDS, tune
from test_evaluate import EVENTS, PRODUCTS, pipeline


class TrackingTests(unittest.TestCase):
    def test_flatten_nested_params(self):
        self.assertEqual(flatten({'alpha': 0.5, 'lightgcn': {'layers': 2}}), {'alpha': 0.5, 'lightgcn.layers': 2})

    def test_jsonl_tracker_appends(self):
        with tempfile.TemporaryDirectory() as d:
            t = JsonlTracker(Path(d)/'runs.jsonl')
            t.log('svd/0', {'factors': 8}, {'dev_ndcg_at_10': 0.1}, {'stage': 'grid'})
            t.log('svd/1', {'factors': 16}, {'dev_ndcg_at_10': 0.2})
            rows = [json.loads(l) for l in (Path(d)/'runs.jsonl').read_text().splitlines()]
        self.assertEqual([r['params'] for r in rows], [{'factors': 8}, {'factors': 16}])


class TuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        pipeline(root, PRODUCTS, EVENTS)
        cls.out = root/'tuning'
        cls.result = tune(root/'splits', cls.out, epochs=2, seeds=(0, 1), n_boot=50)
        cls.run_dir = cls.out/'runs'/cls.result['tuning_run_id']
        cls.trials = [json.loads(l) for l in (cls.run_dir/'trials.jsonl').read_text().splitlines()]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def grid_trials(self, model):
        return [t for t in self.trials if t['tags']['stage'] == 'grid' and t['tags']['model'] == model]

    def test_every_grid_entry_is_tried_and_budget_is_capped(self):
        for model, grid in GRIDS.items():
            self.assertLessEqual(len(grid), 9)
            self.assertEqual(len(self.grid_trials(model)), len(grid))

    def test_selection_is_best_dev_ndcg_with_earliest_tie(self):
        selected = json.loads((self.run_dir/'selected.json').read_text())['models']
        for model in GRIDS:
            trials = self.grid_trials(model)
            best = max(t['metrics']['dev_ndcg_at_10'] for t in trials)
            first_best = next(t for t in trials if t['metrics']['dev_ndcg_at_10'] == best)
            self.assertEqual(selected[model]['params'], first_best['params'])

    def test_hybrids_reuse_selected_components(self):
        selected = json.loads((self.run_dir/'selected.json').read_text())['models']
        for hybrid in ('fixed_hybrid', 'adaptive_hybrid'):
            self.assertEqual(selected[hybrid]['params']['content'], selected['content']['params'])
            self.assertEqual(selected[hybrid]['params']['lightgcn'], selected['lightgcn']['params'])

    def test_seed_repeats_and_ablations_recorded(self):
        stages = {(t['tags']['stage'], t['tags']['model']) for t in self.trials}
        for model in ('lightgcn', 'fixed_hybrid', 'adaptive_hybrid'):
            self.assertIn(('seed', model), stages)
        self.assertIn(('ablation', 'content_no_categories'), stages)
        self.assertEqual(len(self.result['seed_summary']['lightgcn']['dev_ndcg_at_10']), 2)

    def test_never_touches_test_window(self):
        self.assertEqual(self.result['phase'], 'dev')
        self.assertFalse(any(self.out.rglob('test_runs.jsonl')))
        self.assertEqual((self.out/'ACTIVE').read_text().strip(), self.result['tuning_run_id'])
        self.assertTrue((self.run_dir/'summary.md').read_text().startswith('# Dev results'))


if __name__ == '__main__':
    unittest.main()
