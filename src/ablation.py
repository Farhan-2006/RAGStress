def ablation(retriever, classifier, claim, evidence, query, method='hybrid', k=5):
    supporters = {}
    for row in evidence:
        if row['label'] == 'SUPPORT':
            current = supporters.get(row['source_id'])
            if current is None or row['support'] > current['support']:
                supporters[row['source_id']] = row
    sources = sorted(supporters, key=lambda s: -supporters[s]['support'])[:3]
    tests = []
    for source in sources:
        # Remove the source and its exact normalized duplicate group from the entire search space.
        group = supporters[source]['source_group']
        import hashlib
        from .preprocessing import tokenize
        excluded = {doc_id for doc_id, doc in retriever.corpus.items()
                    if hashlib.sha256(' '.join(tokenize(doc['text'])).encode()).hexdigest()[:16] == group}
        excluded.add(source)
        hits = retriever.retrieve(query, k, method, excluded)
        claim_hits = retriever.retrieve(claim, k, method, excluded)
        pooled = {(h['source_id'], h['chunk_id']): h for h in hits + claim_hits}
        rows = classifier.classify(claim, list(pooled.values()))
        survived = any(row['label'] == 'SUPPORT' for row in rows)
        tests.append({'removed_source': source, 'excluded_source_ids': sorted(excluded),
                      'survived': survived, 'reretrieved': hits, 'claim_reretrieved': claim_hits,
                      'evidence': rows})
    return {'tests': tests, 'survival': sum(row['survived'] for row in tests) / len(tests) if tests else None,
            'tested_sources': len(tests), 'total_supporting_sources': len(supporters),
            'coverage_limited': len(supporters) > 3}
