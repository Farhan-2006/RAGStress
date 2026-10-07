"""Inspectable dictionary + postings BM25 with heap Top-K."""
from collections import Counter, defaultdict
import heapq
import math
from .preprocessing import tokenize

class BM25:
    def __init__(self, corpus, k1=1.2, b=0.75):
        self.corpus, self.k1, self.b = corpus, k1, b
        self.postings = defaultdict(dict)
        self.lengths = {}
        for source, document in corpus.items():
            terms = Counter(tokenize(document['title'] + ' ' + document['text']))
            self.lengths[source] = sum(terms.values())
            for term, tf in terms.items():
                self.postings[term][source] = tf
        self.n = len(corpus)
        self.avgdl = sum(self.lengths.values()) / max(1, self.n)
        self.idf = {term: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
                    for term, p in self.postings.items()}

    def retrieve(self, query, k=5, excluded=None):
        excluded = set(excluded or ())
        scores = defaultdict(float)
        for term in sorted(set(tokenize(query))):
            for source, tf in self.postings.get(term, {}).items():
                if source in excluded:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.lengths[source] / max(1, self.avgdl))
                scores[source] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        ranked = heapq.nsmallest(k, scores, key=lambda source: (-scores[source], source))
        return [{'source_id': source, 'score': scores[source], 'bm25': scores[source],
                 'rank': i + 1} for i, source in enumerate(ranked)]

    def inspect(self, query):
        return [{'term': term, 'df': len(self.postings.get(term, {})), 'idf': self.idf.get(term, 0),
                 'postings_sample': list(self.postings.get(term, {}).items())[:8]}
                for term in sorted(set(tokenize(query)), key=lambda t: -self.idf.get(t, 0))]

class Tfidf:
    def __init__(self, corpus):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.ids = list(corpus)
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                        lowercase=False, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([corpus[i]['title'] + ' ' + corpus[i]['text'] for i in self.ids])

    def retrieve(self, query, k=5, excluded=None):
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        excluded = set(excluded or ())
        ranked = heapq.nsmallest(k, (i for i, s in enumerate(scores) if s > 0 and self.ids[i] not in excluded),
                                key=lambda i: (-scores[i], self.ids[i]))
        return [{'source_id': self.ids[i], 'score': float(scores[i]), 'tfidf': float(scores[i]), 'rank': r + 1}
                for r, i in enumerate(ranked)]
