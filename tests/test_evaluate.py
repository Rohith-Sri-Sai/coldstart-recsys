import json
from pathlib import Path
import tempfile
import unittest
from recsys.evaluate import evaluate, evaluate_release, metrics
from recsys.ingest import ingest
from recsys.split import build_split

EPOCH = {'Date First Available': 'January 1, 1970'}


def dump(p, records):
    p.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf8')


def pipeline(root, products, events, cutoffs=(2500,4500,6000)):
    """products: (id, title, details-or-None); events: (user, product, t)."""
    dump(root/'products.jsonl',[{'parent_asin':p,'title':t,**({'details':d} if d else {})} for p,t,d in products])
    dump(root/'reviews.jsonl',[{'user_id':u,'parent_asin':p,'timestamp':t,'rating':5,'text':str(i)} for i,(u,p,t) in enumerate(events)])
    ingest(root/'reviews.jsonl',root/'products.jsonl',root/'ing')
    m=build_split(root/'ing',root/'splits',*cutoffs)
    return root/'splits'/'runs'/m['split_run_id']/'split.sqlite'


PRODUCTS=[('a','red soap',EPOCH),('b','red soap shampoo',EPOCH),('c','blue keyboard',EPOCH),('d','red conditioner',EPOCH)]
EVENTS=[('u1','a',1000),('u1','a',1500),('u2','c',1200),('u1','b',3000),('u1','d',5000),('u2','d',5100)]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.db=pipeline(self.root,PRODUCTS,EVENTS)

    def test_metric_formula(self):
        r,n=metrics(['b','a'],{'a'},2)
        self.assertEqual(r,1)
        self.assertAlmostEqual(n,1/1.584962500721156)

    def test_content_finds_cold_item_and_filters_seen(self):
        r=evaluate(self.db,k=2,phase='dev')
        self.assertEqual(r.rankings['content']['u1'][0],'b')
        self.assertNotIn('a',r.rankings['content']['u1'])
        self.assertEqual(r.report['models']['content']['cohorts']['cold']['recall_at_k'],1)

    def test_popularity_zero_for_cold_candidates(self):
        r=evaluate(self.db,k=1,phase='dev')
        self.assertEqual(r.rankings['popularity']['u1'],['c'])
        self.assertEqual(r.report['models']['popularity']['cohorts']['warm']['users'],0)
        self.assertIsNone(r.report['models']['popularity']['cohorts']['warm']['ndcg_at_k'])

    def test_test_phase_fits_on_everything_before_dev_end(self):
        r=evaluate(self.db,k=3,phase='test')
        # u1 reviewed b during dev: known at the test cutoff, so it is seen, not a candidate.
        self.assertNotIn('b',r.rankings['content']['u1'])
        self.assertEqual(r.report['fit_events_before_ms'],4500)

    def test_ineligible_products_are_never_ranked_or_fitted(self):
        # 'f*' products have no availability date and no prior reviews: not in the catalog.
        extra=[('f1','soap refill',None),('f2','soap pump',None),('f3','soap travel',None)]
        root2=self.root/'second'; root2.mkdir()
        with_extra=evaluate(pipeline(root2,PRODUCTS+extra,EVENTS),k=3,phase='dev')
        base=evaluate(self.db,k=3,phase='dev')
        self.assertEqual(with_extra.rankings,base.rankings)  # TF-IDF vocabulary/IDF unaffected
        self.assertEqual(with_extra.report['models'],base.report['models'])

    def test_intervals_comparisons_and_cold_exposure(self):
        cfg={'models':{'pop':{'model':'popularity'},'content':{'model':'content'}},'compare':[['content','pop']]}
        r=evaluate(self.db,k=2,phase='dev',config=cfg,n_boot=200)
        cold=r.report['models']['content']['cohorts']['cold']
        lo,hi=cold['ndcg_ci95']
        self.assertLessEqual(lo,cold['ndcg_at_k']); self.assertGreaterEqual(hi,cold['ndcg_at_k'])
        self.assertIn('cold',r.report['comparisons']['content - pop'])
        self.assertGreater(r.report['models']['content']['cold_slot_share'],0)

    def test_empty_cohort_and_invalid_args(self):
        r=evaluate(self.db,k=1,phase='dev')
        self.assertIsNone(r.report['models']['content']['cohorts']['sparse']['recall_at_k'])
        with self.assertRaises(ValueError): evaluate(self.db,phase='train')
        with self.assertRaises(ValueError): evaluate(self.db,k=0)

    def test_failure_preserves_active_release(self):
        original=evaluate_release(self.root/'splits',self.root/'eval',k=2)['evaluation_run_id']
        with self.assertRaises(RuntimeError):evaluate_release(self.root/'splits',self.root/'eval',k=2,fail_before_commit=True)
        self.assertEqual((self.root/'eval'/'ACTIVE').read_text().strip(),original)

    def test_every_test_run_is_logged(self):
        evaluate_release(self.root/'splits',self.root/'eval',k=2,phase='dev')
        self.assertFalse((self.root/'eval'/'test_runs.jsonl').exists())
        for _ in range(2): evaluate_release(self.root/'splits',self.root/'eval',k=2,phase='test')
        lines=(self.root/'eval'/'test_runs.jsonl').read_text().splitlines()
        self.assertEqual(len(lines),2)
        self.assertEqual(json.loads(lines[0])['config_sha256'],json.loads(lines[1])['config_sha256'])


if __name__=='__main__':unittest.main()
