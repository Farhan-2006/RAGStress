import re

def variants(query):
    match = re.fullmatch(r'Does (.+?) (reduce|increase|improve|prevent) (.+?)\??', query.strip().rstrip('?'), re.IGNORECASE)
    if match:
        subject, verb, target = match.groups()
        forms = {'reduce': ('reduced', 'reduces'), 'increase': ('increased', 'increases'),
                 'improve': ('improved', 'improves'), 'prevent': ('prevented', 'prevents')}
        passive, present = forms[verb.casefold()]
        return [f'Is {target} {passive} by {subject}?',
                f'{target}: is it {passive} by {subject}?',
                f'What is the evidence on whether {subject} {present} {target}?']
    # Controlled wording probes, not automatically validated semantic paraphrases.
    core = query.strip().rstrip('?.')
    return [f'What does the scientific evidence say about {core}?',
            f'Research evidence regarding {core}', f'Find studies relevant to {core}']

def jaccard(left, right):
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a | b else 0.0

def stability(retriever, query, method='hybrid', k=5, rewrites=None):
    base = retriever.retrieve(query, k, method)
    ids = [h['source_id'] for h in base]
    rows = []
    for rewrite in (rewrites if rewrites is not None else variants(query)):
        hits = retriever.retrieve(rewrite, k, method)
        other = [h['source_id'] for h in hits]
        rows.append({'query': rewrite, 'source_ids': other, 'jaccard': jaccard(ids, other)})
    kind = ('user-supplied rewrites' if rewrites is not None else
            'rule-based active/passive question rewrites (review for equivalence)' if query.strip().lower().startswith('does ') and any(v in query.lower() for v in [' reduce ', ' increase ', ' improve ', ' prevent ']) else
            'template wording perturbations')
    return {'kind': kind,
            'base_ids': ids, 'variants': rows,
            'mean_jaccard': sum(row['jaccard'] for row in rows) / len(rows) if rows else 0.0}
