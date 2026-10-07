"""Pure presentation helpers; no retrieval or trust decisions live here."""
import hashlib
import html
import json
import re

DECISIONS = {'KEEP': ('✓', 'Supported under these tests'),
             'QUALIFY': ('!', 'Answer needs qualification'),
             'REWRITE': ('↻', 'Answer corrected'),
             'ABSTAIN': ('?', 'Insufficient evidence')}

def source_catalog(result):
    """Stable source labels across initial retrieval and counter-search."""
    catalog = {}
    def visit(value):
        if isinstance(value, dict):
            if value.get('source_id') is not None:
                sid = str(value['source_id'])
                item = catalog.setdefault(sid, {'number': len(catalog) + 1, 'id': sid,
                    'title': f'Document {sid}', 'passages': [], 'quotes': []})
                if value.get('title'):
                    item['title'] = value['title']
                for field, target in [('passage', 'passages'), ('quote', 'quotes')]:
                    if value.get(field) and value[field] not in item[target]:
                        item[target].append(value[field])
            for key, child in value.items():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(result.get('retrieved', []))
    visit(result)
    return catalog

def source_label(sid, catalog):
    item = catalog.get(str(sid))
    return f"Source {item['number']} · ID {sid}" if item else f'Unmapped ID {sid}'

def citations(text, catalog):
    def replace(match):
        sid = match[1]
        return f"[Source {catalog[sid]['number']}; ID {sid}]" if sid in catalog else f'[unmapped reference: {sid}]'
    return re.sub(r'\[([A-Za-z0-9_-]+)\]', replace, text)


def answer_paragraph(result):
    """Keep evidence meaning, moving all document bookkeeping out of prose."""
    text = result['final_answer']
    text = re.sub(r'Supporting:\s*(?:\[[^\]]+\]\s*)*;\s*conflicting:\s*(?:\[[^\]]+\]\s*)*',
                  'Both supporting and conflicting evidence were found. ', text)
    text = re.sub(r'Source\s*\[[^\]]+\]\s+adds:', 'Additional evidence indicates:', text)
    text = re.sub(r'Source\s*\[[^\]]+\]\s+reports:', 'The evidence reports:', text)
    text = re.sub(r'\[(?:Source \d+; ID )?[A-Za-z0-9_-]+\]', '', text)
    return ' '.join(text.split())


def approved_source_documents(result):
    """Only evidence actually used for retained content or a correction qualifies."""
    if result.get('decision') == 'ABSTAIN':
        return []
    referenced = set(re.findall(r'\[([A-Za-z0-9_-]+)\]',
                     result.get('answer_with_references', result['final_answer'])))
    catalog = source_catalog(result)
    approved = {}
    for index, claim in enumerate(result.get('claims', []), 1):
        if claim['decision'] == 'ABSTAIN':
            continue
        correction = claim['decision'] == 'REWRITE'
        for evidence in claim.get('evidence', []):
            sid = str(evidence['source_id'])
            labels = ('CONTRADICT',) if correction else ('SUPPORT', 'WEAK_SUPPORT')
            if sid not in referenced or evidence['label'] not in labels:
                continue
            item = approved.setdefault(sid, {'source_id':sid, 'title':catalog[sid]['title'],
                'number':catalog[sid]['number'], 'roles':[], 'claim_numbers':[], 'quotes':[]})
            role = 'Evidence for the correction' if correction else 'Qualified supporting evidence' if claim['decision']=='QUALIFY' else 'Supporting evidence'
            for field, value in [('roles',role), ('claim_numbers',index), ('quotes',evidence['quote'])]:
                if value not in item[field]:
                    item[field].append(value)
    return sorted(approved.values(), key=lambda d:d['number'])

def highlighted_quote(text, query):
    terms = set(re.findall(r'[\w-]+', query.casefold()))
    pieces = re.split(r'([\w-]+)', text)
    return ''.join('<mark>' + html.escape(p) + '</mark>' if p.casefold() in terms else html.escape(p)
                   for p in pieces).replace('\n', '<br>')

def settings_snapshot(query, claim_mode, method, generator, k, rewrites, thresholds, remote_identity=''):
    # Fingerprint detects endpoint/model changes, without exporting endpoint or key.
    return {'query': query.strip(), 'claim_mode': claim_mode, 'method': method,
            'generator': generator if not claim_mode else 'user-claim', 'k': k,
            'rewrites': rewrites, 'thresholds': thresholds,
            'remote_config_fingerprint': hashlib.sha256(remote_identity.encode()).hexdigest() if generator == 'remote' and not claim_mode else None}

def result_is_stale(previous, current):
    return previous is not None and previous != current

def evidence_summary(result):
    claims = result.get('claims', [])
    support = [r for c in claims for r in c.get('evidence', []) if r['label'] in ('SUPPORT','WEAK_SUPPORT')]
    counter = [r for c in claims for r in c.get('evidence', []) if r['label'] == 'CONTRADICT']
    return {'support_documents': len({r['source_id'] for r in support}),
            'support_passages': len({(r['source_id'], r['quote']) for r in support}),
            'support_groups': len({r['source_group'] for r in support}),
            'counter_documents': len({r['source_id'] for r in counter}),
            'counter_passages': len({(r['source_id'], r['quote']) for r in counter}),
            'unverified': result.get('stance_model') is None,
            'claim_decisions': [c['decision'] for c in claims]}

def ranking_rows(hits, catalog):
    return [{'Rank': h['rank'], 'Source': source_label(h['source_id'], catalog),
             'Title': h.get('title', ''), 'BM25 score': h.get('bm25'),
             'Dense cosine': h.get('dense'), 'RRF score': h.get('rrf'),
             'TF-IDF cosine': h.get('tfidf')} for h in hits]

def claim_reason(claim):
    status = claim['status']
    components = claim.get('components', {})
    return {'UNVERIFIED': 'The evidence-checking model is unavailable. Retrieval similarity cannot verify a claim.',
            'CONTESTED': 'The model found both supporting and conflicting passages. The disagreement needs to remain visible.',
            'CONTRADICTED': 'The model classified retrieved passages as conflicting with this claim. The claim is not retained; the final assessment attributes the retrieved evidence.',
            'UNSTABLE': 'The retrieved sources change substantially with the tested wording variants.',
            'ROBUST_WITHIN_CORPUS': 'Strong relevant evidence supports the claim, with no confirmed contradiction or severe wording instability.'}.get(status, claim['reason'])

def decision_reason(result):
    reasons = list(dict.fromkeys(claim_reason(c) for c in result.get('claims', [])))
    if result.get('answer_repair'):
        return 'The generated wording failed verification and was replaced by an attributed corpus quotation. ' + ' '.join(reasons)
    return ' '.join(reasons) or 'No auditable claim was produced from the retrieved evidence.'

def safe_export(result):
    """Drop credentials/configuration paths, retaining exact scientific evidence."""
    forbidden = {'key', 'api_key', 'password', 'token', 'secret', 'endpoint', 'url', 'path',
                 'local_path', 'cache_path', 'remote_config_fingerprint'}
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k.casefold() not in forbidden}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    exported = clean(result)
    if isinstance(exported, dict) and 'final_answer' in exported and 'claims' in exported:
        documents = approved_source_documents(exported)
        exported['answer_with_references'] = exported.get('answer_with_references', exported['final_answer'])
        exported['final_answer'] = answer_paragraph(exported)
        exported['approved_source_documents'] = documents
    return exported

def markdown_audit(result):
    result = safe_export(result)
    catalog = source_catalog(result)
    lines = ['# RAGStress evidence audit', '', f"Input: {result['query']}",
             f"Mode: {'Check a claim' if result.get('claim_mode') else 'Ask a question'}",
             f"Retrieval: {result['method']} · Top-K: {result['k']}", '',
             '## Final answer', answer_paragraph(result), '',
             f"Decision: {result['decision']} — {DECISIONS[result['decision']][1]}",
             decision_reason(result), '', '## Original draft or supplied claim', result['initial_answer'],
             'Draft citations are provisional; mapping a source does not establish support.', '',
             '## Submitted settings', '```json', json.dumps(result.get('run_settings', {}), indent=2), '```', '',
             f"Actual generator: {result['generation_mode']}", f"Stance model: {result.get('stance_model') or 'Unavailable'}",
             'Model revisions: ' + json.dumps(result.get('model_revisions', {})), '']
    lines.extend(['## Approved source documents'])
    documents = approved_source_documents(result)
    if not documents:
        lines.append('No source documents were approved for an answer.')
    for document in documents:
        lines.append(f"Source {document['number']} · ID {document['source_id']}: {document['title']} — {', '.join(document['roles'])}")
    if result.get('answer_repair'):
        lines.extend(['## Correction', result['answer_repair']['reason'], result['answer_repair']['quote']])
    if result.get('superseded_claims'):
        lines.append('## Rejected generated claims')
        for claim in result['superseded_claims']:
            lines.extend([claim['claim'], f"{claim['decision']}: {claim['reason']}"])
    for i, claim in enumerate(result.get('claims', []), 1):
        lines.extend(['', f'## Claim {i}', claim['claim'], f"{claim['decision']} / {claim['status']}: {claim_reason(claim)}"])
        for label in ['SUPPORT', 'WEAK_SUPPORT', 'CONTRADICT', 'NEUTRAL']:
            rows = [r for r in claim.get('evidence', []) if r['label'] == label]
            lines.append(f'### {label} ({len(rows)} passages)')
            for row in rows:
                lines.extend([source_label(row['source_id'], catalog), '> ' + row['quote'].replace('\n', '\n> ')])
        lines.append('### Counter-search queries')
        lines.extend(r['query'] for r in claim.get('counter_evidence', {}).get('runs', []))
    lines.extend(['', '## Wording stability', 'Original: ' + result['query'], result['stability']['kind'],
                  'Original sources: ' + ', '.join(result['stability']['base_ids'])])
    for v in result['stability']['variants']:
        lines.append(f"{v['query']} — Jaccard {v['jaccard']:.3f}; sources: {', '.join(v['source_ids'])}")
    lines.extend(['', '## All audited sources'])
    for item in catalog.values():
        lines.append(f"Source {item['number']} · ID {item['id']}: {item['title']}")
    lines.extend(['', '## Limitations',
                  'Heuristic audit of a bounded corpus; not a calibrated truth probability.',
                  'Distinct IDs and duplicate-aware groups do not prove independent studies.',
                  'Neural stance labels can be wrong. Low wording overlap does not imply a false answer.',
                  'Sentence/semicolon claim candidates can contain multiple assertions. Human evaluation is pending.'])
    return '\n\n'.join(lines)
