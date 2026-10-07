import hashlib
import json
import logging
import numpy as np
from .config import CACHE, DENSE_MODEL, MODEL_REVISIONS
from .preprocessing import chunks

class Dense:
    def __init__(self, corpus, model_name=DENSE_MODEL):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(4)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device='cpu', revision=MODEL_REVISIONS[model_name])
        self.windows = [window for document in corpus.values() for window in chunks(document)]
        texts = [w['title'] + ' ' + w['text'] for w in self.windows]
        payload = json.dumps({'model': model_name, 'revision': MODEL_REVISIONS[model_name], 'max_seq_length': self.model.max_seq_length,
                              'windows': self.windows}, sort_keys=True).encode()
        self.fingerprint = hashlib.sha256(payload).hexdigest()
        path = CACHE / f'dense-{self.fingerprint[:16]}.npy'
        if path.exists():
            self.matrix = np.load(path, allow_pickle=False)
        else:
            logging.info('Encoding %d sentence windows', len(texts))
            self.matrix = self.model.encode(texts, batch_size=48, normalize_embeddings=True,
                                            show_progress_bar=True).astype('float32')
            np.save(path, self.matrix)
        if self.matrix.shape[0] != len(self.windows):
            raise ValueError('Embedding cache mismatch')
        self.query_cache = {}

    def encode_query(self, query):
        if query not in self.query_cache:
            self.query_cache[query] = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        return self.query_cache[query]

    def retrieve(self, query, k=5, excluded=None):
        excluded = set(excluded or ())
        scores = self.matrix @ self.encode_query(query)
        # Max pooling over windows, then ranking unique sources. Same source is never multiple evidence votes.
        best = {}
        for i, window in enumerate(self.windows):
            source = window['source_id']
            if source in excluded:
                continue
            if source not in best or scores[i] > best[source][0]:
                best[source] = (float(scores[i]), i)
        ranked = sorted(best, key=lambda source: (-best[source][0], source))[:k]
        return [{'source_id': source, 'score': best[source][0], 'dense': best[source][0],
                 'chunk_id': self.windows[best[source][1]]['id'],
                 'passage': self.windows[best[source][1]]['text'], 'rank': r + 1}
                for r, source in enumerate(ranked)]
