"""A separate pretrained NLI model audits corpus sentences, not its own generation."""
import hashlib
import re
import numpy as np
from .config import (NLI_MODEL, NLI_THRESHOLD, MODEL_REVISIONS, WEAK_SUPPORT_THRESHOLD,
    CONTRADICTION_THRESHOLD, TOPICAL_COVERAGE_THRESHOLD, SEMANTIC_ALIGNMENT_THRESHOLD)
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
        self.embedding_cache = {}

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
        contexts = self.predict([(hit['passage'], claim) for hit, _ in candidates])
        dense = self.retriever.dense
        semantic = [None] * len(candidates)
        if dense is not None:
            if not hasattr(self, 'embedding_cache'):
                self.embedding_cache = {}
            missing = list(dict.fromkeys(s for s in [claim] + [s for _, s in candidates] if s not in self.embedding_cache))
            if missing:
                vectors = dense.model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
                self.embedding_cache.update(zip(missing, vectors))
            semantic = [float(self.embedding_cache[claim] @ self.embedding_cache[s]) for _, s in candidates]
        rows = []
        entity_terms = {term.casefold().strip('-') for term in re.findall(r'\b(?:[A-Z][A-Z0-9-]{2,}|[A-Za-z]+[0-9][A-Za-z0-9-]*)\b', claim)}
        total_idf = sum(self.retriever.sparse.idf.get(term, 0) for term in query_terms)
        aliases = {'tp53':['p53'], 'ppm1d':['wip1'], 'hiv':['human immunodeficiency virus'],
                   'art':['antiretroviral therapy'], 'dna':['deoxyribonucleic acid'], 'rna':['ribonucleic acid']}
        for (hit, sentence), values, context, cosine in zip(candidates, scores, contexts, semantic):
            contradiction, support, neutral = map(float, values)
            label = 'NEUTRAL'
            passage_terms = set(tokenize(sentence))
            passage_terms.update(part for term in list(passage_terms) for part in re.split(r'[.-]', term))
            missing_entities = sorted(e for e in entity_terms if e not in passage_terms
                and not any(alias in sentence.casefold() for alias in aliases.get(e, []))
                and not (not any(c.isdigit() for c in e) and any(re.fullmatch(re.escape(e)+r'\d+', t) for t in passage_terms)))
            topical_coverage = sum(self.retriever.sparse.idf.get(term, 0) for term in query_terms & passage_terms) / total_idf if total_idf else 0.0
            # Generic NLI can confidently compare unrelated p53 passages to a PPM1D claim.
            # Reject decisive labels when a named alphanumeric entity is absent from the premise window.
            hard_mismatch = any(any(c.isdigit() for c in entity) for entity in missing_entities)
            aligned = topical_coverage >= TOPICAL_COVERAGE_THRESHOLD or (cosine is not None and cosine >= SEMANTIC_ALIGNMENT_THRESHOLD)
            # Generic acronym absence is secondary to strong semantic alignment; specific gene mismatches remain blocking.
            generic_ok = not missing_entities or (cosine is not None and cosine >= .50 and support >= NLI_THRESHOLD)
            if aligned and not hard_mismatch and generic_ok:
                if support >= NLI_THRESHOLD:
                    label = 'SUPPORT'
                elif support >= WEAK_SUPPORT_THRESHOLD:
                    label = 'WEAK_SUPPORT'
                elif contradiction >= CONTRADICTION_THRESHOLD and float(context[0]) >= CONTRADICTION_THRESHOLD and (topical_coverage >= .25 or (cosine is not None and cosine >= .50)):
                    label = 'CONTRADICT'
            document = self.retriever.corpus[hit['source_id']]
            normalized = ' '.join(tokenize(document['text']))
            rows.append({'source_id': hit['source_id'], 'chunk_id': hit['chunk_id'],
                         'quote': sentence, 'label': label, 'support': support,
                         'contradiction': contradiction, 'neutral': neutral,
                         'missing_entity_anchors': missing_entities,
                         'hard_entity_mismatch': hard_mismatch, 'semantic_alignment': cosine,
                         'context_contradiction': float(context[0]),
                         'alignment_passed': aligned, 'entity_check_passed': not hard_mismatch and generic_ok,
                         'idf_weighted_claim_term_coverage': topical_coverage,
                         'source_group': hashlib.sha256(normalized.encode()).hexdigest()[:16],
                         'retrieval_rank': hit['rank']})
        return rows
