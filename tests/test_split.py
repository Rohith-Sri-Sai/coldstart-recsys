import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from recsys.ingest import ingest
from recsys.split import build_split

EPOCH = {'Date First Available': 'January 1, 1970'}  # available before every fixture cutoff


def dump(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')


def run(root, events, products, cutoffs=(2500, 4000, 6000)):
    ing, out = root/'ing', root/'out'
    dump(root/'reviews.jsonl',[{'user_id':u,'parent_asin':p,'timestamp':t,'rating':5,'text':str(i)} for i,(u,p,t) in enumerate(events)])
    dump(root/'products.jsonl',products)
    ingest(root/'reviews.jsonl',root/'products.jsonl',ing)
    m=build_split(ing,out,*cutoffs)
    return m, out/'runs'/m['split_run_id']/'split.sqlite'


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        # u1 has history; p2 first reviewed during dev, p3 during test.
        # u2 has no history before dev -> not a dev target user.
        self.events=[('u1','p1',1000),('u1','p1',2000),('u1','p2',3000),('u2','p2',3200),('u1','p3',5000)]
        self.products=[{'parent_asin':p,'title':p,'details':EPOCH} for p in ('p1','p2','p3','p4')]

    def test_boundaries_and_eligible_user(self):
        m,_=run(self.root,self.events,self.products)
        self.assertEqual((m['stats']['train'],m['stats']['dev'],m['stats']['test']),(2,1,1))
        self.assertEqual(m['stats']['dev_label_cohorts'],{'cold':1})
        self.assertEqual(m['stats']['test_label_cohorts'],{'cold':1})

    def test_events_after_test_end_are_ignored(self):
        m,_=run(self.root,self.events+[('u1','p4',7000)],self.products)
        self.assertEqual(m['stats']['test'],1)
        self.assertEqual(m['stats']['ignored_after_test_end'],1)

    def test_cohorts_do_not_use_future_events(self):
        _,db=run(self.root,self.events,self.products)
        with closing(sqlite3.connect(db)) as c:
            self.assertEqual(c.execute("SELECT cohort FROM phase_cohorts WHERE phase='dev' AND product_id='p2'").fetchone()[0],'cold')
            self.assertEqual(c.execute("SELECT cohort FROM phase_cohorts WHERE phase='test' AND product_id='p2'").fetchone()[0],'sparse')

    def test_cohort_counts_distinct_users_not_events(self):
        _,db=run(self.root,self.events,self.products)
        with closing(sqlite3.connect(db)) as c:
            self.assertEqual(c.execute("SELECT prior_users FROM phase_cohorts WHERE phase='dev' AND product_id='p1'").fetchone()[0],1)

    def test_catalog_eligibility_uses_only_pre_cutoff_information(self):
        products=[{'parent_asin':'old','title':'t','details':EPOCH},                      # listed, never reviewed
                  {'parent_asin':'nodate','title':'t'},                                  # no date, no reviews
                  {'parent_asin':'later','title':'t','details':{'Date First Available':'January 1, 2030'}},
                  {'parent_asin':'reviewed','title':'t'}]                                # no date, reviewed in train
        events=[('u1','reviewed',1000),('u1','nodate',3000),('u1','later',3100)]
        m,db=run(self.root,events,products)
        with closing(sqlite3.connect(db)) as c:
            got=dict(c.execute("SELECT product_id, eligible FROM phase_cohorts WHERE phase='dev'"))
        self.assertEqual(got,{'old':1,'nodate':0,'later':0,'reviewed':1})
        # Reviews inside the window must NOT make an item eligible: that would leak the label.
        self.assertEqual(m['stats']['dev_labels_dropped_ineligible'],2)
        self.assertEqual(m['stats']['dev_label_cohorts'],{})

    def test_first_available_after_first_review_is_counted(self):
        products=[{'parent_asin':'p','title':'t','details':{'Date First Available':'January 1, 2030'}}]
        m,_=run(self.root,[('u1','p',1000)],products)
        self.assertEqual(m['stats']['first_available_after_first_review'],1)

    def test_bad_cutoffs_rejected(self):
        for cut in ((4000,2500,6000),(2500,4000,4000)):
            with self.assertRaises(ValueError): run(self.root,self.events,self.products,cut)

    def test_failure_preserves_active(self):
        m,_=run(self.root,self.events,self.products)
        with self.assertRaises(RuntimeError): build_split(self.root/'ing',self.root/'out',2500,4000,6000,fail_before_commit=True)
        self.assertEqual((self.root/'out'/'ACTIVE').read_text().strip(),m['split_run_id'])

if __name__=='__main__': unittest.main()
