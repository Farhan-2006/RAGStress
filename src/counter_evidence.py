from .preprocessing import tokenize

def counter_queries(claim):
    core = claim.strip().rstrip('.?')
    return [core, core + ' conflicting evidence', core + ' no significant effect ineffective',
            core + ' refuted does not contrary evidence']

def counter_evidence(retriever, classifier, claim, method='hybrid', k=5):
    queries = counter_queries(claim)
    pooled, runs = {}, []
    for query in queries:
        hits = retriever.retrieve(query, k, method)
        runs.append({'query': query, 'hits': hits})
        for hit in hits:
            pooled.setdefault((hit['source_id'], hit['chunk_id']), hit)
    # Bound the audit. Prefer original-claim ranking first; all retrieved runs remain visible.
    candidates = list(pooled.values())[:15]
    rows = classifier.classify(claim, candidates) if classifier else []
    return {'queries': queries, 'runs': runs, 'evidence': rows, 'audited_passages': len(candidates)}
