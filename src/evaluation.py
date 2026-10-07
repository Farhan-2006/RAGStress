import math
import numpy as np

def metrics(ranked, qrels, k=5):
    ids = list(dict.fromkeys(ranked))[:k]
    relevant = {doc for doc, grade in qrels.items() if grade > 0}
    found = sum(doc in relevant for doc in ids)
    dcg = sum((2 ** qrels.get(doc, 0) - 1) / math.log2(rank + 2) for rank, doc in enumerate(ids))
    ideal = sum((2 ** grade - 1) / math.log2(rank + 2)
                for rank, grade in enumerate(sorted((g for g in qrels.values() if g > 0), reverse=True)[:k]))
    precision, recall = found/k, found/len(relevant) if relevant else 0
    return {f'P@{k}': precision, f'Recall@{k}': recall,
            f'F1@{k}': 2*precision*recall/(precision+recall) if precision+recall else 0,
            f'MRR@{k}': next((1 / (i + 1) for i, doc in enumerate(ids) if doc in relevant), 0),
            f'nDCG@{k}': dcg / ideal if ideal else 0}

def evaluate(retriever, queries, qrels, method, k=5):
    rows = []
    runs = {}
    for qid in sorted(qrels, key=lambda x: int(x)):
        hits = retriever.retrieve(queries[qid], k, method)
        ids = [h['source_id'] for h in hits]
        runs[qid] = hits
        rows.append({'qid': qid, **metrics(ids, qrels[qid], k)})
    means = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != 'qid'}
    return {'method': method, 'queries': len(rows), **means}, rows, runs

def bootstrap_difference(baseline, final, metric, seed=358, repetitions=2000):
    """Paired query bootstrap; estimate uncertainty, do not tune on held-out test."""
    base = {row['qid']: row[metric] for row in baseline}
    diffs = np.array([row[metric] - base[row['qid']] for row in final])
    rng = np.random.default_rng(seed)
    values = diffs[rng.integers(0, len(diffs), (repetitions, len(diffs)))].mean(axis=1)
    low, high = np.quantile(values, [0.025, 0.975])
    return {'metric': metric, 'mean_delta': float(diffs.mean()), 'ci95': [float(low), float(high)]}
