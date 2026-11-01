import gzip
import json
from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path
from recsys.ingest import ingest


def write(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write((row if isinstance(row, str) else json.dumps(row)) + '\n')


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.reviews = self.root / 'reviews.jsonl'
        self.products = self.root / 'products.jsonl'
        self.out = self.root / 'output'
        self.review = {'user_id': 'u1', 'parent_asin': 'p1', 'timestamp': 1600000000000, 'rating': 5.0, 'text': 'good'}
        write(self.reviews, [self.review])
        write(self.products, [{'parent_asin': 'p1', 'title': 'Soap', 'features': ['mild']}])

    def tearDown(self):
        self.temp.cleanup()

    def test_commit_pointer_and_join(self):
        manifest = ingest(self.reviews, self.products, self.out)
        self.assertEqual((self.out / 'ACTIVE').read_text().strip(), manifest['run_id'])
        self.assertEqual(manifest['counts']['events_with_metadata'], 1)

    def test_invalid_records_are_quarantined(self):
        write(self.reviews, [self.review, '{broken', {'user_id': '', 'parent_asin': 'p1', 'timestamp': 1, 'rating': 5}])
        manifest = ingest(self.reviews, self.products, self.out)
        self.assertEqual(manifest['counts']['reviews_quarantined'], 2)
        path = self.out / 'runs' / manifest['run_id'] / 'quarantine.jsonl'
        self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_duplicates_are_deterministic(self):
        write(self.reviews, [self.review, self.review])
        manifest = ingest(self.reviews, self.products, self.out)
        self.assertEqual(manifest['counts']['reviews_accepted'], 1)
        self.assertEqual(manifest['counts']['reviews_duplicate'], 1)

    def test_failed_build_does_not_replace_active(self):
        first = ingest(self.reviews, self.products, self.out)['run_id']
        with self.assertRaises(RuntimeError):
            ingest(self.reviews, self.products, self.out, fail_before_commit=True)
        self.assertEqual((self.out / 'ACTIVE').read_text().strip(), first)

    def test_gzip_reviews(self):
        compressed = self.root / 'reviews.jsonl.gz'
        with gzip.open(compressed, 'wt', encoding='utf-8') as f:
            f.write(json.dumps(self.review) + '\n')
        manifest = ingest(compressed, self.products, self.out)
        self.assertEqual(manifest['counts']['reviews_accepted'], 1)

    def test_invalid_timestamp_bool(self):
        write(self.reviews, [{**self.review, 'timestamp': True}])
        manifest = ingest(self.reviews, self.products, self.out)
        self.assertEqual(manifest['counts']['reviews_quarantined'], 1)

    def product_rows(self, products):
        write(self.products, products)
        manifest = ingest(self.reviews, self.products, self.out)
        db = self.out / 'runs' / manifest['run_id'] / 'records.sqlite'
        with closing(sqlite3.connect(db)) as conn:
            rows = dict(conn.execute('SELECT product_id, available_from_ms FROM products'))
        return manifest, rows

    def test_first_available_date_parsed(self):
        # Raw JSONL has details as a dict; the HF parquet copy has it as a JSON string.
        manifest, rows = self.product_rows([
            {'parent_asin': 'p1', 'title': 'Soap', 'details': {'Date First Available': 'March 1, 2018'}},
            {'parent_asin': 'p2', 'title': 'Gel', 'details': json.dumps({'Date First Available': 'Jan 5, 2020'})},
            {'parent_asin': 'p3', 'title': 'Oil', 'details': {'Date First Available': 'sometime soon'}},
            {'parent_asin': 'p4', 'title': 'Tea'}])
        self.assertEqual(rows, {'p1': 1519862400000, 'p2': 1578182400000, 'p3': None, 'p4': None})
        self.assertEqual(manifest['counts']['products_with_first_available'], 2)
        self.assertEqual(manifest['counts']['products_first_available_unparsed'], 1)

    def test_categories_kept_apart_from_content(self):
        self.product_rows([{'parent_asin': 'p1', 'title': 'Soap', 'features': ['mild'],
                            'main_category': 'All Beauty', 'categories': ['Skin', 'Face']}])
        with closing(sqlite3.connect(next(self.out.glob('runs/*/records.sqlite')))) as conn:
            content, categories = conn.execute('SELECT content, categories FROM products').fetchone()
        self.assertEqual(content, 'Soap mild')
        self.assertEqual(categories, 'All Beauty Skin Face')

    def test_crawl_time_fields_never_reach_content(self):
        # average_rating, rating_number, price and bought_together are 2023 crawl values.
        self.product_rows([{'parent_asin': 'p1', 'title': 'Soap', 'average_rating': 4.7,
                            'rating_number': 98765, 'price': 12.34, 'bought_together': ['zzbt']}])
        with closing(sqlite3.connect(next(self.out.glob('runs/*/records.sqlite')))) as conn:
            content = conn.execute('SELECT content FROM products').fetchone()[0]
        for leaked in ('4.7', '98765', '12.34', 'zzbt'):
            self.assertNotIn(leaked, content)


if __name__ == '__main__':
    unittest.main()
