"""A separate pretrained NLI model audits corpus sentences, not its own generation."""
import hashlib
import re
import numpy as np
from .config import NLI_MODEL, NLI_THRESHOLD, MODEL_REVISIONS
from .preprocessing import sentences, tokenize

class Stance:
    def __init__(self, retriever, model_name=NLI_MODEL):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        torch.set_num_threads(4)
        self.retriever = retriever
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=MODEL_REVISIONS[model_name])
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=MODEL_REVISIONS[model_name]).eval()
        # Fixed order is explicitly documented by THIS model card.
        if model_name not in ('cross-encoder/nli-MiniLM2-L6-H768', 'cross-encoder/nli-deberta-v3-small'):
            raise ValueError('Configure and verify label mapping before changing NLI model.')
        self.cache = {}

    def predict(self, pairs):
        import torch
        # Group short premises together to reduce padding and CPU cost; scores map back by pair.
        missing = sorted(dict.fromkeys(pair for pair in pairs if pair not in self.cache),
                         key=lambda pair: len(pair[0]) + len(pair[1]))
        for start in range(0, len(missing), 16):
            batch = missing[start:start + 16]
            encoded = self.tokenizer([p[0] for p in batch], [p[1] for p in batch],
                                     padding=True, truncation=True, max_length=384, return_tensors='pt')
            with torch.inference_mode():
                probabilities = self.model(**encoded).logits.softmax(dim=-1).cpu().numpy()
            for pair, scores in zip(batch, probabilities):
                self.cache[pair] = scores
        return np.array([self.cache[pair] for pair in pairs])

    def classify(self, claim, hits):
        candidates = []
        query_terms = set(tokenize(claim))
        for hit in hits:
            # Actual displayed passage only. Never classify hidden gold rationales at runtime.
            choices = sentences(hit['passage'])
            choices = sorted(choices, key=lambda s: -sum(self.retriever.sparse.idf.get(t, 0)
                             for t in query_terms & set(tokenize(s))))[:3]
            for sentence in choices:
                candidates.append((hit, sentence))
            # Preserve context for evidence that needs more than one sentence.
            if hit['passage'] not in choices:
                candidates.append((hit, hit['passage']))
        if not candidates:
            return []
        scores = self.predict([(sentence, claim) for _, sentence in candidates])
        rows = []
        entity_terms = {term.casefold().strip('-') for term in re.findall(r'\b(?:[A-Z][A-Z0-9-]{2,}|[A-Za-z]+[0-9][A-Za-z0-9-]*)\b', claim)}
        total_idf = sum(self.retriever.sparse.idf.get(term, 0) for term in query_terms)
        for (hit, sentence), values in zip(candidates, scores):
            contradiction, support, neutral = map(float, values)
            label = 'SUPPORT' if support >= NLI_THRESHOLD else 'CONTRADICT' if contradiction >= NLI_THRESHOLD else 'NEUTRAL'
            passage_terms = set(tokenize(hit['passage']))
            passage_terms.update(part for term in list(passage_terms) for part in re.split(r'[.-]', term))
            missing_entities = sorted(entity_terms - passage_terms)
            topical_coverage = sum(self.retriever.sparse.idf.get(term, 0) for term in query_terms & passage_terms) / total_idf if total_idf else 0.0
            # Generic NLI can confidently compare unrelated p53 passages to a PPM1D claim.
            # Reject decisive labels when a named alphanumeric entity is absent from the premise window.
            if missing_entities or topical_coverage < 0.25:
                label = 'NEUTRAL'
            document = self.retriever.corpus[hit['source_id']]
            normalized = ' '.join(tokenize(document['text']))
            rows.append({'source_id': hit['source_id'], 'chunk_id': hit['chunk_id'],
                         'quote': sentence, 'label': label, 'support': support,
                         'contradiction': contradiction, 'neutral': neutral,
                         'missing_entity_anchors': missing_entities,
                         'idf_weighted_claim_term_coverage': topical_coverage,
                         'source_group': hashlib.sha256(normalized.encode()).hexdigest()[:16],
                         'retrieval_rank': hit['rank']})
        return rows
