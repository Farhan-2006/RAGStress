import time
from .ablation import ablation
from .claims import extract_claims
from .counter_evidence import counter_evidence
from .stability import stability
from .trust import decide
from .config import NLI_MODEL, NLI_THRESHOLD, STABILITY_THRESHOLD, MODEL_REVISIONS
from .answer_repair import repair_candidates

class Pipeline:
    def __init__(self, retriever, generator, classifier=None):
        self.retriever, self.generator, self.classifier = retriever, generator, classifier

    def _audit(self, claim, hits, query, method, k, query_test, relevant):
        claim_hits = self.retriever.retrieve(claim, k, method)
        direct = self.classifier.classify(claim, hits + claim_hits) if self.classifier else []
        counter = counter_evidence(self.retriever, self.classifier, claim, method, k)
        evidence = list({(row['source_id'], row['quote']): row for row in direct + counter['evidence']}.values())
        source_test = ablation(self.retriever, self.classifier, claim, evidence, query, method, k) if self.classifier else {'tests': [], 'survival': None}
        trust = decide(evidence, source_test, query_test, self.classifier is not None, relevant)
        return {'claim': claim, 'claim_retrieval': claim_hits, 'evidence': evidence,
                'counter_evidence': counter, 'ablation': source_test, **trust}

    def run(self, query, method='hybrid', k=5, claim_mode=False, rewrites=None):
        if not query.strip():
            raise ValueError('Please enter a question or claim.')
        started = time.perf_counter()
        hits = self.retriever.retrieve(query, k, method)
        if claim_mode:
            answer, mode, warning = query, 'user-claim', None
        else:
            answer, mode, warning = self.generator.generate(query, hits)
        claims, truncated = extract_claims(answer)
        query_test = stability(self.retriever, query, method, k, rewrites)
        audits, final = [], []
        # Explicit prototype relevance gate. Cosine is NOT stance or calibrated correctness.
        if self.retriever.dense is not None:
            dense_hits = self.retriever.dense.retrieve(query, 1)
            query_score = dense_hits[0]['dense'] if dense_hits else 0
            relevant = query_score >= 0.30
        else:
            terms = self.retriever.sparse.inspect(query)
            query_score = None
            relevant = any(term['df'] > 0 for term in terms)
        superseded, repair = [], None
        audits = [self._audit(claim, hits, query, method, k, query_test, relevant) for claim in claims]
        if not claim_mode and relevant and self.classifier is not None and (not audits or all(a['decision'] == 'ABSTAIN' for a in audits)):
            for candidate in repair_candidates(self.retriever, query, hits):
                replacement = self._audit(candidate['quote'], hits + [candidate['hit']], query, method, k, query_test, relevant)
                if replacement['decision'] in ('KEEP', 'QUALIFY') and any(row['label'] == 'SUPPORT' for row in replacement['evidence']):
                    superseded, audits = audits, [replacement]
                    repair = {'reason': 'Generated wording failed verification; replaced with an attributed finding from a ranked corpus source.',
                              'source_id': candidate['hit']['source_id'], 'chunk_id': candidate['hit']['chunk_id'],
                              'quote': candidate['quote'], 'query_cosine': candidate['query_cosine'],
                              'query_term_coverage': candidate['query_term_coverage'],
                              'evidence_passage': candidate['hit']['passage']}
                    truncated = False
                    break
        for audit in audits:
            claim, evidence, trust = audit['claim'], audit['evidence'], audit
            rows = [row for row in evidence if row['label'] == ('CONTRADICT' if trust['decision'] == 'REWRITE' else 'SUPPORT')]
            best = max(rows, key=lambda row: row['contradiction'] if trust['decision'] == 'REWRITE' else row['support'], default=None)
            ids = sorted({row['source_id'] for row in rows})
            references = ' '.join(f'[{source}]' for source in ids)
            if trust['decision'] == 'ABSTAIN':
                final.append('I could not retrieve sufficiently robust evidence from this corpus to answer this claim confidently.')
            elif trust['decision'] == 'REWRITE':
                final.append(f"The proposed claim is not retained. Source [{best['source_id']}] reports: \"{best['quote']}\"")
            elif trust['status'] == 'CONTESTED':
                counter_ids = sorted({row['source_id'] for row in evidence if row['label'] == 'CONTRADICT'})
                final.append(f'Corpus evidence is contested for: {claim} Supporting: {references}; conflicting: ' + ' '.join(f'[{s}]' for s in counter_ids))
            else:
                prefix = 'Evidence is fragile or wording-sensitive: ' if trust['decision'] == 'QUALIFY' else ''
                final.append(prefix + claim + ' ' + references)
        if not audits:
            final = ['I could not retrieve sufficiently robust evidence from this corpus to answer this confidently.']
        if truncated:
            final.append('Additional generated sentences were omitted because they were not audited.')
        decisions = [a['decision'] for a in audits]
        global_decision = ('ABSTAIN' if not decisions or all(d == 'ABSTAIN' for d in decisions)
                           else 'QUALIFY' if 'ABSTAIN' in decisions or 'QUALIFY' in decisions
                           else 'REWRITE' if 'REWRITE' in decisions else 'KEEP')
        if repair:
            global_decision = 'REWRITE'
            final = [f"The generated answer was replaced with retrieved evidence. Source [{repair['source_id']}] reports: \"{repair['quote']}\"",
                     f"Evidence audit: {audits[0]['status']}. {audits[0]['reason']}"]
        return {'query': query, 'method': method, 'k': k, 'initial_answer': answer, 'generation_mode': mode,
                'model_revisions': MODEL_REVISIONS, 'stance_model': NLI_MODEL if self.classifier else None,
                'prototype_thresholds': {'stance': NLI_THRESHOLD, 'query_jaccard': STABILITY_THRESHOLD, 'query_cosine': 0.30, 'claim_term_coverage': 0.25},
                'warning': warning, 'retrieved': hits, 'query_terms': self.retriever.sparse.inspect(query),
                'answer_repair': repair, 'superseded_claims': superseded,
                'query_relevance_cosine': query_score, 'stability': query_test, 'claims': audits,
                'claim_extraction': 'sentence/semicolon candidates; complex sentences may not be atomic',
                'unexamined_sentences_omitted': truncated, 'decision': global_decision,
                'final_answer': '\n\n'.join(final), 'elapsed_seconds': time.perf_counter() - started}
