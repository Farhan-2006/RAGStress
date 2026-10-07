import argparse
import csv
import json
import logging
import platform
import time
from pathlib import Path
from .config import RESULTS, SEED, NLI_MODEL, DENSE_MODEL, GENERATOR_MODEL
from .data import prepare, load_corpus, load_queries, load_qrels, jsonl
from .sparse_retriever import BM25, Tfidf
from .hybrid_retriever import Retriever

def build(dense=True):
    corpus = load_corpus()
    sparse, tfidf = BM25(corpus), Tfidf(corpus)
    encoder = None
    if dense:
        from .dense_retriever import Dense
        encoder = Dense(corpus)
    return Retriever(corpus, sparse, encoder, tfidf)

def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf8')

def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def retrieval_experiment(args):
    from .evaluation import evaluate, bootstrap_difference
    retriever = build(not args.sparse_only)
    queries, qrels = load_queries(), load_qrels(args.split)
    methods = ['tfidf', 'bm25'] + ([] if args.sparse_only else ['dense', 'hybrid'])
    summaries, rows_by_method = [], {}
    for method in methods:
        started = time.perf_counter()
        summary, rows, runs = evaluate(retriever, queries, qrels, method, args.k)
        summary['seconds'] = time.perf_counter() - started
        summaries.append(summary)
        rows_by_method[method] = rows
        write_csv(RESULTS / f'{args.split}-{method}-per-query.csv', rows)
        write_json(RESULTS / f'{args.split}-{method}-run.json', runs)
        logging.info('%s', summary)
    differences = []
    if 'hybrid' in methods:
        differences = [bootstrap_difference(rows_by_method['bm25'], rows_by_method['hybrid'], metric)
                       for metric in (f'P@{args.k}', f'Recall@{args.k}', f'MRR@{args.k}', f'nDCG@{args.k}')]
    payload = {'split': args.split, 'k': args.k, 'corpus_sources': len(retriever.corpus),
               'dense_windows': len(retriever.dense.windows) if retriever.dense else None,
               'summary': summaries, 'hybrid_minus_bm25': differences,
               'protocol': 'Macro averages, unique source IDs, all split qrels, binary relevance; no test tuning.',
               'seed': SEED, 'python': platform.python_version(),
               'dense_model': DENSE_MODEL if retriever.dense else None}
    write_json(RESULTS / f'retrieval-{args.split}.json', payload)
    write_csv(RESULTS / f'retrieval-{args.split}.csv', summaries)
    print(json.dumps(payload, indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for axis, metric in zip(axes, [f'Recall@{args.k}', f'nDCG@{args.k}']):
            axis.bar([s['method'] for s in summaries], [s[metric] for s in summaries], color=['#8794a8', '#356fa2', '#60aaa2', '#7659a0'][:len(summaries)])
            axis.set_ylim(0, 1)
            axis.set_title(metric)
        fig.suptitle(f'SciFact {args.split}: {len(qrels)} queries / {len(retriever.corpus)} sources')
        fig.tight_layout()
        fig.savefig(RESULTS / f'retrieval-{args.split}.png', dpi=180)
        plt.close(fig)
    except ImportError:
        logging.warning('Plotting package unavailable; CSV/JSON saved.')

def novelty_experiment(args):
    import random
    from .stability import stability
    from .counter_evidence import counter_queries
    retriever = build(not args.sparse_only)
    queries, qrels = load_queries(), load_qrels('test')
    rng = random.Random(SEED)
    sampled = rng.sample(sorted(qrels), min(args.limit, len(qrels)))
    methods = ['bm25'] + ([] if args.sparse_only else ['dense', 'hybrid'])
    stability_rows, counter_rows = [], []
    from .config import DATA
    gold = {str(row['id']): row for row in jsonl(DATA / 'original/claims_dev.jsonl')}
    for method in methods:
        for qid in sampled:
            query = queries[qid]
            probe = stability(retriever, query, method, args.k)
            stability_rows.append({'method': method, 'qid': qid, 'jaccard': probe['mean_jaccard']})
            hits = retriever.retrieve(query, args.k, method)
            base_ids = [h['source_id'] for h in hits]
            labels = gold.get(qid, {}).get('evidence', {})
            contradictions = {source for source, annotation in labels.items() if any(a['label'] == 'CONTRADICT' for a in annotation)}
            if contradictions:
                after = {h['source_id'] for variant in counter_queries(query)
                         for h in retriever.retrieve(variant, args.k, method)}
                equal_budget = {h['source_id'] for h in retriever.retrieve(query, 4 * args.k, method)}
                counter_rows.append({'method': method, 'qid': qid, 'gold_contradicting_sources': len(contradictions),
                                     'initial_contradiction_recall': len(set(base_ids) & contradictions) / len(contradictions),
                                     'expanded_contradiction_recall': len(after & contradictions) / len(contradictions),
                                     'budget_matched_original_recall': len(equal_budget & contradictions) / len(contradictions),
                                     'expanded_unique_sources': len(after)})
    write_csv(RESULTS / 'stability.csv', stability_rows)
    write_csv(RESULTS / 'counter-retrieval-gold.csv', counter_rows)
    summary = []
    for method in methods:
        stability_values = [r['jaccard'] for r in stability_rows if r['method'] == method]
        counters = [r for r in counter_rows if r['method'] == method]
        summary.append({'method': method, 'queries': len(sampled),
                        'mean_jaccard': sum(stability_values) / len(stability_values),
                        'contradiction_queries': len(counters),
                        'initial_counter_recall': sum(r['initial_contradiction_recall'] for r in counters) / len(counters) if counters else None,
                        'expanded_counter_recall': sum(r['expanded_contradiction_recall'] for r in counters) / len(counters) if counters else None,
                        'budget_matched_original_counter_recall': sum(r['budget_matched_original_recall'] for r in counters) / len(counters) if counters else None})
    write_json(RESULTS / 'novelty.json', {'seed': SEED, 'sampled_test_qids': sampled, 'summary': summary,
                                        'limitations': 'Counter expansion retrieves up to 4K slots versus K initial; recall gain is not an equal-budget comparison. Template probes are not verified paraphrases. Gold used only for evaluation. No gold alternative is not evidence of claim falsity.'})
    print(json.dumps(summary, indent=2))

def stance_experiment(args):
    from .stance import Stance
    from .config import DATA
    retriever = build(False)
    classifier = Stance(retriever)
    corpus = {str(row['doc_id']): row for row in jsonl(DATA / 'original/corpus.jsonl')}
    original_split = 'train' if args.split == 'train' else 'dev'
    dev = jsonl(DATA / f'original/claims_{original_split}.jsonl')
    import random
    random.Random(SEED).shuffle(dev)
    pairs, metadata = [], []
    for claim in dev:
        for source, rationales in claim['evidence'].items():
            for rationale in rationales[:1]:
                premise = ' '.join(corpus[source]['abstract'][index] for index in rationale['sentences'])
                pairs.append((premise, claim['claim']))
                metadata.append({'qid': str(claim['id']), 'source': source, 'gold': rationale['label']})
        if len(pairs) >= args.limit:
            break
    scores = classifier.predict(pairs)
    from .config import NLI_THRESHOLD
    rows = []
    for row, score in zip(metadata, scores):
        label = 'SUPPORT' if score[1] >= NLI_THRESHOLD else 'CONTRADICT' if score[0] >= NLI_THRESHOLD else 'NEUTRAL'
        rows.append({**row, 'predicted': label, 'correct': row['gold'] == label,
                     'support': float(score[1]), 'contradiction': float(score[0])})
    slug = NLI_MODEL.rsplit('/', 1)[-1]
    write_csv(RESULTS / f'stance-{slug}-{original_split}.csv', rows)
    summary = {'pairs': len(rows), 'accuracy': sum(r['correct'] for r in rows) / len(rows),
               'coverage': sum(r['predicted'] != 'NEUTRAL' for r in rows) / len(rows),
               'accuracy_on_decisive_pairs': sum(r['correct'] for r in rows) / max(1, sum(r['predicted'] != 'NEUTRAL' for r in rows)),
               'threshold': NLI_THRESHOLD, 'model': NLI_MODEL,
               'split': original_split,
               'protocol': 'Seeded original annotated rationale pairs only; gold supplied as premise for isolated classifier diagnosis, never runtime. Not end-to-end accuracy; no neutral gold here.'}
    write_json(RESULTS / f'stance-{slug}-{original_split}.json', summary)
    write_json(RESULTS / 'stance.json', summary)
    print(json.dumps(summary, indent=2))

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description='RAGStress real-corpus retrieval and audits')
    parser.add_argument('command', choices=['download', 'evaluate', 'novelty', 'stance', 'ask', 'warmup'])
    parser.add_argument('--sparse-only', action='store_true')
    parser.add_argument('--split', choices=['train', 'test'], default='test')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--query', default='What potential does antiretroviral therapy have to prevent HIV-associated tuberculosis?')
    parser.add_argument('--method', choices=['bm25', 'tfidf', 'dense', 'hybrid'], default='hybrid')
    parser.add_argument('--generator', choices=['local', 'remote', 'extractive'], default='local')
    parser.add_argument('--claim-mode', action='store_true')
    parser.add_argument('--output', default='results/latest-audit.json')
    args = parser.parse_args()
    if args.k < 1 or args.limit < 1:
        parser.error('k and limit must be positive')
    if args.command == 'download':
        print(json.dumps(prepare(), indent=2))
    elif args.command == 'evaluate':
        retrieval_experiment(args)
    elif args.command == 'novelty':
        novelty_experiment(args)
    elif args.command == 'stance':
        stance_experiment(args)
    else:
        retriever = build(not args.sparse_only)
        from .generator import Generator
        from .stance import Stance
        from .pipeline import Pipeline
        generator = Generator(args.generator)
        try:
            classifier = Stance(retriever)
        except Exception as error:
            classifier = None
            logging.warning('NLI unavailable (%s); claims will remain unverified.', type(error).__name__)
        result = Pipeline(retriever, generator, classifier).run(args.query, args.method, args.k, args.claim_mode)
        write_json(args.output, result)
        print(result['decision'] + '\n' + result['final_answer'])
        print(f"Generation: {result['generation_mode']}; runtime: {result['elapsed_seconds']:.1f}s; audit: {args.output}")

if __name__ == '__main__':
    main()
