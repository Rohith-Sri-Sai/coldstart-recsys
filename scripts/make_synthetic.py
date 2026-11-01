"""Write a synthetic dataset in Amazon Reviews 2023 format, for smoke tests only.

Items belong to topics, users prefer one or two topics, and users only review
items already listed. Results on this data say nothing about real quality; it
only checks that the pipeline runs end to end and how long it takes.

python scripts/make_synthetic.py --out data/synthetic --users 20000 --items 5000 --reviews 120000
"""
import argparse
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import random

TOPICS = ['shampoo', 'serum', 'lipstick', 'perfume', 'nail', 'razor', 'sunscreen', 'mascara', 'soap', 'brush']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('data/synthetic'))
    p.add_argument('--users', type=int, default=20000)
    p.add_argument('--items', type=int, default=5000)
    p.add_argument('--reviews', type=int, default=120000)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    rng = random.Random(a.seed)
    start, end = datetime(2018, 1, 1, tzinfo=timezone.utc), datetime(2023, 9, 1, tzinfo=timezone.utc)
    span = (end - start).total_seconds()
    a.out.mkdir(parents=True, exist_ok=True)
    items = []
    with gzip.open(a.out/'meta.jsonl.gz', 'wt', encoding='utf8') as f:
        for n in range(a.items):
            topic = rng.choice(TOPICS)
            listed = start + timedelta(seconds=rng.random() * span * 0.95)
            item = {'parent_asin': f'S{n:07d}', 'title': f'{topic} {rng.choice(["mini", "pro", "daily", "gentle"])} {n}',
                    'features': [f'{topic} care', rng.choice(['vegan', 'travel size', 'unscented', 'value pack'])],
                    'main_category': 'All Beauty', 'categories': [topic]}
            if rng.random() < 0.7:  # roughly the coverage to expect; check the real number
                item['details'] = {'Date First Available': listed.strftime('%B %d, %Y')}
            f.write(json.dumps(item) + '\n')
            items.append((item['parent_asin'], topic, listed.timestamp()))
    by_topic = {t: [i for i in items if i[1] == t] for t in TOPICS}
    tastes = [rng.sample(TOPICS, rng.choice([1, 2])) for _ in range(a.users)]
    weights = [rng.paretovariate(1.2) for _ in range(a.users)]  # a few heavy reviewers, many light ones
    with gzip.open(a.out/'reviews.jsonl.gz', 'wt', encoding='utf8') as f:
        written = 0
        while written < a.reviews:
            u = rng.choices(range(a.users), weights)[0] if written % 4 else rng.randrange(a.users)
            t = start.timestamp() + rng.random() * span
            topic = rng.choice(tastes[u]) if rng.random() < 0.85 else rng.choice(TOPICS)
            listed = [i for i in by_topic[topic] if i[2] <= t]
            if not listed:
                continue
            # Popular-ish skew: earlier-listed items get picked a bit more often.
            pid = listed[min(int(rng.expovariate(1 / max(1, len(listed) / 4))), len(listed) - 1)][0]
            f.write(json.dumps({'user_id': f'U{u:06d}', 'parent_asin': pid, 'asin': pid, 'rating': rng.choice([3, 4, 5, 5, 5]),
                                'timestamp': int(t * 1000), 'text': f'review {written}'}) + '\n')
            written += 1


if __name__ == '__main__':
    main()
