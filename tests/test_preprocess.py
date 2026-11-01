import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from recsys.preprocess import preprocess, audit_run


class PreprocessTests(unittest.TestCase):
    def test_raw_fixture_audit_and_provenance(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / 'ingestion'
            report = preprocess(root/'data/fixtures/reviews.jsonl', root/'data/fixtures/products.jsonl', output)
            self.assertEqual(report['accepted_events'], report['counts']['reviews_accepted'])
            self.assertEqual(report['products'], report['counts']['products_accepted'])
            self.assertEqual(report['events_without_product_metadata'],
                             report['accepted_events'] - report['events_with_product_metadata'])
            run = output/'runs'/report['ingestion_run_id']
            self.assertTrue((run/'audit.json').is_file())
            self.assertEqual(json.loads((run/'audit.json').read_text())['input_checksums'], report['input_checksums'])
            self.assertEqual(audit_run(output)['accepted_events'], report['accepted_events'])

    def test_missing_raw_input_does_not_change_active_run(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmp:
            output = Path(tmp)/'ingestion'
            preprocess(root/'data/fixtures/reviews.jsonl', root/'data/fixtures/products.jsonl', output)
            active = (output/'ACTIVE').read_text()
            with self.assertRaises(FileNotFoundError):
                preprocess(root/'data/fixtures/does-not-exist.jsonl', root/'data/fixtures/products.jsonl', output)
            self.assertEqual((output/'ACTIVE').read_text(), active)


if __name__ == '__main__':
    unittest.main()
