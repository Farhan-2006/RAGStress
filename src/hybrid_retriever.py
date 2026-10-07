from .config import POOL_K, RRF_CONSTANT
from .preprocessing import chunks, tokenize

def rrf(sparse, dense, k=5, constant=RRF_CONSTANT):
    merged = {}
    for field, hits in [('sparse_rank', sparse), ('dense_rank', dense)]:
        for rank, hit in enumerate(hits, 1):
            source = hit['source_id']
            row = merged.setdefault(source, {'source_id': source, 'score': 0.0})
            row['score'] += 1 / (constant + rank)
            row[field] = rank
            for key, value in hit.items():
                if key not in ('rank', 'score'):
                    row[key] = value
    ranked = sorted(merged.values(), key=lambda row: (-row['score'], row['source_id']))[:k]
    return [dict(row, rank=i + 1, rrf=row['score']) for i, row in enumerate(ranked)]

class Retriever:
    def __init__(self, corpus, sparse, dense=None, tfidf=None):
        self.corpus, self.sparse, self.dense, self.tfidf = corpus, sparse, dense, tfidf

    def retrieve(self, query, k=5, method='hybrid', excluded=None):
        if method in ('dense', 'hybrid') and self.dense is None:
            raise RuntimeError('Dense model unavailable; explicitly select bm25 or tfidf.')
        if method == 'hybrid':
            hits = rrf(self.sparse.retrieve(query, POOL_K, excluded),
                       self.dense.retrieve(query, POOL_K, excluded), k)
        elif method == 'bm25':
            hits = self.sparse.retrieve(query, k, excluded)
        elif method == 'tfidf' and self.tfidf is not None:
            hits = self.tfidf.retrieve(query, k, excluded)
        elif method == 'dense':
            hits = self.dense.retrieve(query, k, excluded)
        else:
            raise ValueError(f'Unknown/unavailable method {method}')
        terms = set(tokenize(query))
        for hit in hits:
            document = self.corpus[hit['source_id']]
            hit['title'] = document['title']
            if 'passage' not in hit:
                candidates = chunks(document)
                window = max(candidates, key=lambda w: sum(self.sparse.idf.get(t, 0) for t in terms & set(tokenize(w['text']))))
                hit['passage'], hit['chunk_id'] = window['text'], window['id']
        return hits
