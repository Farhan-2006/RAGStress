import time
from difflib import SequenceMatcher
from .claims import extract_claims
from .counter_evidence import counter_evidence
from .stability import stability
from .trust import decide
from .config import NLI_MODEL, NLI_THRESHOLD, STABILITY_THRESHOLD, MODEL_REVISIONS
from .answer_repair import repair_candidates

class Pipeline:
    def __init__(self, retriever, generator, classifier=None, enable_counter=True, enable_stability=True):
        self.retriever, self.generator, self.classifier = retriever, generator, classifier
        self.enable_counter, self.enable_stability = enable_counter, enable_stability

    def _audit(self, claim, hits, query, method, k, query_test, relevant, progress=None, timings=None):
        started = time.perf_counter()
        if progress:
            progress('Checking claims')
        claim_hits = self.retriever.retrieve(claim, k, method)
        direct = self.classifier.classify(claim, hits + claim_hits) if self.classifier else []
        direct_seconds = time.perf_counter() - started
        started = time.perf_counter()
        if progress:
            progress('Searching for counter-evidence')
        counter = counter_evidence(self.retriever, self.classifier, claim, method, k) if self.enable_counter else {'runs': [], 'evidence': [], 'queries': [], 'audited_passages': 0}
        counter_seconds = time.perf_counter() - started
        if timings is not None:
            timings['verification'] += direct_seconds
            timings['counter_evidence'] += counter_seconds
        evidence = list({(row['source_id'], row['quote']): row for row in direct + counter['evidence']}.values())
        trust = decide(evidence, query_test, self.classifier is not None, relevant)
        return {'claim': claim, 'claim_retrieval': claim_hits, 'evidence': evidence,
                'counter_evidence': counter, 'timings': {'verification': direct_seconds, 'counter_evidence': counter_seconds}, **trust}

    def run(self, query, method='hybrid', k=5, claim_mode=False, rewrites=None, progress=None):
        if not query.strip():
            raise ValueError('Please enter a question or claim.')
        started = time.perf_counter()
        timings = {'verification':0.0,'counter_evidence':0.0}
        stage_started = time.perf_counter()
        if progress:
            progress('Retrieving sources')
        hits = self.retriever.retrieve(query, k, method)
        timings['retrieval'] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        if claim_mode:
            answer, mode, warning = query, 'user-claim', None
        else:
            if progress:
                progress('Generating the draft')
            answer, mode, warning = self.generator.generate(query, hits)
        timings['generation'] = time.perf_counter() - stage_started
        claims, truncated = extract_claims(answer)
        if progress:
            progress('Testing wording stability')
        stage_started = time.perf_counter()
        query_test = stability(self.retriever, query, method, k, rewrites) if self.enable_stability else {'kind':'not tested','tested':False,'base_ids':[],'variants':[],'mean_jaccard':None}
        timings['query_stability'] = time.perf_counter() - stage_started
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
        audits = [self._audit(claim, hits, query, method, k, query_test, relevant, progress, timings) for claim in claims]
        for audit in audits:
            audit['origin'] = 'user_claim' if claim_mode else 'generated_draft'
        if not claim_mode and relevant and self.classifier is not None and (not audits or all(a['decision'] == 'ABSTAIN' for a in audits)):
            for candidate in repair_candidates(self.retriever, query, hits):
                replacement = self._audit(candidate['quote'], hits + [candidate['hit']], query, method, k, query_test, relevant, progress, timings)
                if replacement['decision'] in ('KEEP', 'QUALIFY') and any(row['label'] == 'SUPPORT' for row in replacement['evidence']):
                    superseded, audits = audits, [replacement]
                    replacement['origin'] = 'replacement_quote'
                    repair = {'reason': 'Generated wording failed verification; replaced with an attributed finding from a ranked corpus source.',
                              'source_id': candidate['hit']['source_id'], 'chunk_id': candidate['hit']['chunk_id'],
                              'quote': candidate['quote'], 'query_cosine': candidate['query_cosine'],
                              'query_term_coverage': candidate['query_term_coverage'],
                              'evidence_passage': candidate['hit']['passage']}
                    truncated = False
                    break
        # A tiny local generator often ignores paragraph instructions. Add real,
        # relevant corpus detail only after the same claim audit accepts it.
        context_details = []
        if not claim_mode and relevant and self.classifier is not None and any(a['decision'] in ('KEEP', 'QUALIFY') for a in audits):
            for candidate in repair_candidates(self.retriever, query, hits, limit=6):
                kept = [a['claim'] for a in audits if a['decision'] in ('KEEP', 'QUALIFY')]
                if sum(len(c.split()) for c in kept) >= 65 or len(context_details) >= 2 or len(audits) >= 6:
                    break
                text = ' '.join(candidate['quote'].split())
                if any(SequenceMatcher(None, text.casefold(), c.casefold()).ratio() >= .80 for c in kept):
                    continue
                detail = self._audit(text, hits + [candidate['hit']], query, method, k, query_test, relevant, progress, timings)
                if detail['decision'] in ('KEEP', 'QUALIFY') and any(r['label'] == 'SUPPORT' for r in detail['evidence']):
                    detail['origin'] = 'retrieved_context'
                    detail['context_source_id'] = candidate['hit']['source_id']
                    audits.append(detail)
                    context_details.append({'source_id':candidate['hit']['source_id'], 'quote':text})
        for audit in audits:
            claim, evidence, trust = audit['claim'], audit['evidence'], audit
            rows = [row for row in evidence if row['label'] == 'CONTRADICT'] if trust['decision'] == 'REWRITE' else [row for row in evidence if row['label'] in ('SUPPORT','WEAK_SUPPORT')]
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
                prefix = 'Evidence needs qualification: ' if trust['decision'] == 'QUALIFY' else ''
                if audit.get('origin') == 'retrieved_context':
                    final.append(prefix + f"Source [{audit['context_source_id']}] adds: \"{claim}\"")
                else:
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
            audited_detail_parts = final[1:]
            final = [f"The generated answer was replaced with retrieved evidence. Source [{repair['source_id']}] reports: \"{repair['quote']}\"",
                     f"Evidence audit: {audits[0]['status']}. {audits[0]['reason']}"]
            final.extend(audited_detail_parts)
        if global_decision == 'ABSTAIN':
            final = list(dict.fromkeys(final))
            final.append('The available passages could not be verified as sufficient support. This is an inconclusive corpus assessment, rather than evidence that the proposed claim is false.')
        elif claim_mode:
            final.append(' '.join(dict.fromkeys(a['reason'] for a in audits)))
            final.append('This assessment is limited to the retrieved SciFact evidence and does not establish that all scientific studies agree.')
        if progress:
            progress('Preparing the result')
        elapsed = time.perf_counter() - started
        timings['other_audit'] = elapsed - sum(timings.values())
        return {'query': query, 'method': method, 'k': k, 'initial_answer': answer, 'generation_mode': mode,
                'corpus_sources': len(self.retriever.corpus), 'claim_mode': claim_mode,
                'model_revisions': MODEL_REVISIONS, 'stance_model': NLI_MODEL if self.classifier else None,
                'prototype_thresholds': {'strong_support': NLI_THRESHOLD, 'query_jaccard': STABILITY_THRESHOLD, 'query_cosine': 0.30},
                'warning': warning, 'retrieved': hits, 'query_terms': self.retriever.sparse.inspect(query),
                'answer_repair': repair, 'superseded_claims': superseded,
                'query_relevance_cosine': query_score, 'stability': query_test, 'claims': audits,
                'claim_extraction': 'sentence/semicolon candidates; complex sentences may not be atomic',
                'unexamined_sentences_omitted': truncated, 'decision': global_decision,
                'answer_context': context_details, 'answer_format': 'single_paragraph',
                'final_answer': ' '.join(' '.join(part.split()) for part in final), 'elapsed_seconds': elapsed, 'timings': timings,
                'components_enabled': {'counter_evidence':self.enable_counter,'query_stability':self.enable_stability}}
