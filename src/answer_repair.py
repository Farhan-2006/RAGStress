"""Recover an attributed finding from ranked corpus sources, never model memory."""
import math
import re
from .preprocessing import chunks, sentences, tokenize

QUERY_WRAPPERS = set('what which how does do is are was were can could would should the a an of to in for on by with and or about evidence scientific studies research say tell me please'.split())
FINDING = re.compile(r'conclusion|result|associat|reduc|improv|increas|decreas|prevent|suppress|promot|inhibit|demonstrat|found|showed|suggest|effective|correlat|significan', re.I)
METHOD = re.compile(r'^(?:BACKGROUND|METHODS|OBJECTIVE|We (?:conducted|aimed|investigated|studied)|The aim)', re.I)

def repair_candidates(retriever, query, hits, limit=3):
    """Read sentence windows of already-ranked sources and expose provenance.

    The quote is a corpus finding, not a verification of the rejected answer.
    Query coverage includes unknown content words at maximum IDF so nonsense
    queries cannot pass by matching only common words. Gates are prototype
    retrieval heuristics; final quote still undergoes the ordinary evidence audit.
    """
    terms = set(tokenize(query)) - QUERY_WRAPPERS
    if not terms:
        return []
    unknown_idf = math.log(1 + (retriever.sparse.n + .5) / .5)
    weights = {term: retriever.sparse.idf.get(term, unknown_idf) for term in terms}
    total = sum(weights.values())
    candidates, seen = [], set()
    for hit in hits:
        for window in chunks(retriever.corpus[hit['source_id']]):
            for quote in sentences(window['text']):
                if (hit['source_id'], quote) in seen or not 8 <= len(quote.split()) <= 85:
                    continue
                seen.add((hit['source_id'], quote))
                coverage = sum(value for term, value in weights.items() if term in set(tokenize(quote))) / total
                if coverage < .35:
                    continue
                candidates.append({'quote': quote, 'query_term_coverage': coverage,
                                   'is_finding': bool(FINDING.search(quote)) and not bool(METHOD.search(quote)),
                                   'hit': dict(hit, passage=window['text'], chunk_id=window['id'])})
    if not candidates:
        return []
    if retriever.dense is not None:
        query_vector = retriever.dense.encode_query(query)
        vectors = retriever.dense.model.encode([row['quote'] for row in candidates],
                                               normalize_embeddings=True, show_progress_bar=False)
        for row, vector in zip(candidates, vectors):
            row['query_cosine'] = float(vector @ query_vector)
        candidates = [row for row in candidates if row['query_cosine'] >= .40]
    else:
        candidates = [dict(row, query_cosine=None) for row in candidates if row['query_term_coverage'] >= .60]
    # Findings take precedence over a methodology sentence. No invented confidence score.
    candidates.sort(key=lambda row: (not row['is_finding'], -(row['query_cosine'] or row['query_term_coverage']),
                                     -row['query_term_coverage'], row['hit']['rank']))
    return candidates[:limit]
