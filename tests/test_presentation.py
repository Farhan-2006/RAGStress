import copy
from src.presentation import (source_catalog, citations, highlighted_quote, settings_snapshot,
    result_is_stale, ranking_rows, evidence_summary, safe_export, markdown_audit,
    answer_paragraph, approved_source_documents)
from src.pipeline import Pipeline
from src.sparse_retriever import BM25
from src.hybrid_retriever import Retriever

def audit():
    class Generator:
        def generate(self, query, hits):
            return 'Alpha increases. [1]', 'fixture', None
    corpus = {'1': {'id': '1', 'title': 'Evidence', 'text': 'Alpha increases.'}}
    return Pipeline(Retriever(corpus, BM25(corpus)), Generator()).run('alpha', 'bm25')

def test_stage_callbacks_do_not_change_decisions():
    result = audit()
    stages = []
    class Generator:
        def generate(self, query, hits):
            return 'Alpha increases. [1]', 'fixture', None
    corpus = {'1': {'id': '1', 'title': 'Evidence', 'text': 'Alpha increases.'}}
    actual = Pipeline(Retriever(corpus, BM25(corpus)), Generator()).run('alpha', 'bm25', progress=stages.append)
    assert stages == ['Retrieving sources', 'Generating the draft', 'Testing wording stability',
                      'Checking claims', 'Searching for counter-evidence', 'Preparing the result']
    assert [{k:v for k,v in c.items() if k != 'timings'} for c in actual['claims']] == [{k:v for k,v in c.items() if k != 'timings'} for c in result['claims']]
    assert actual['final_answer'] == result['final_answer']

def test_source_labels_cover_counter_and_claim_documents():
    result = audit()
    result['extra'] = {'hits': [{'source_id': '2', 'title': 'Counter', 'passage': 'Opposite.'}],
                       'claim_hits': [{'source_id':'3','title':'Third study','passage':'A finding.'}]}
    catalog = source_catalog(result)
    assert list(catalog) == ['1', '2', '3']
    assert citations('Evidence [1] [2] [404]', catalog) == 'Evidence [Source 1; ID 1] [Source 2; ID 2] [unmapped reference: 404]'
    assert source_catalog(result) == catalog

def test_highlighting_is_exact_and_html_escaped():
    value = highlighted_quote('Alpha <script>alert(1)</script> alphabet.', 'alpha')
    assert '<script>' not in value and '&lt;script&gt;' in value
    assert '<mark>Alpha</mark>' in value and '<mark>alphabet</mark>' not in value

def test_unavailable_scores_stay_missing():
    result = audit()
    row = ranking_rows(result['retrieved'], source_catalog(result))[0]
    assert row['BM25 score'] is not None
    assert row['Dense cosine'] is None and row['RRF score'] is None

def test_settings_detect_input_mode_and_remote_changes_without_secrets():
    before = settings_snapshot('alpha', False, 'hybrid', 'remote', 5, None, {'stance': .8}, 'secret-endpoint-A')
    after = settings_snapshot('alpha', False, 'hybrid', 'remote', 5, None, {'stance': .8}, 'secret-endpoint-B')
    assert result_is_stale(before, after)
    changed = copy.deepcopy(before)
    changed['claim_mode'] = True
    assert result_is_stale(before, changed)
    assert not result_is_stale(before, copy.deepcopy(before))
    assert 'secret-endpoint' not in str(before)
    assert 'remote_config_fingerprint' not in safe_export(before)

def test_no_model_or_test_is_not_zero_confidence():
    result = audit()
    summary = evidence_summary(result)
    assert summary['unverified'] and summary['support_documents'] == 0

def test_readable_export_contains_exact_run_and_no_configuration_secrets():
    result = audit()
    result['run_settings'] = {'query': 'alpha', 'key': 'do-not-export', 'path': 'C:/private',
                              'remote_config_fingerprint': 'also-private'}
    text = markdown_audit(result)
    assert result['query'] in text and result['initial_answer'] in text
    assert 'ABSTAIN' in text and 'Source 1' in text and 'Model revisions' in text
    assert all(secret not in text for secret in ['do-not-export', 'C:/private', 'also-private'])


def approved_fixture():
    return {'decision':'QUALIFY',
        'final_answer':'Alpha increases. [1] Source [1] adds: "Alpha rises in adults."',
        'retrieved':[{'source_id':str(i), 'title':f'Study {i}'} for i in [1,2,3]],
        'claims':[{'decision':'QUALIFY', 'evidence':[
            {'source_id':'1','label':'SUPPORT','quote':'Alpha rises in adults.'},
            {'source_id':'2','label':'CONTRADICT','quote':'Alpha decreases.'},
            {'source_id':'3','label':'NEUTRAL','quote':'A methodology.'}]}]}


def test_approved_sources_exclude_unrelated_and_conflicting_retrieval_hits():
    result = approved_fixture()
    assert [d['source_id'] for d in approved_source_documents(result)] == ['1']
    text = answer_paragraph(result)
    assert '[' not in text and 'Source' not in text
    assert 'Alpha rises in adults.' in text
    exported = safe_export(result)
    assert exported['final_answer'] == text
    assert approved_source_documents(exported) == approved_source_documents(result)
    result['decision'] = 'ABSTAIN'
    assert approved_source_documents(result) == []


def test_contested_and_corrected_paragraphs_preserve_meaning_without_source_ids():
    contested = {'final_answer':'Evidence is contested. Supporting: [1]; conflicting: [2]'}
    assert 'Both supporting and conflicting evidence' in answer_paragraph(contested)
    assert '[2]' not in answer_paragraph(contested)
    corrected = approved_fixture()
    corrected['decision'] = 'REWRITE'
    corrected['final_answer'] = 'The proposed claim is not retained. Source [2] reports: "Alpha decreases."'
    corrected['claims'][0]['decision'] = 'REWRITE'
    assert [d['source_id'] for d in approved_source_documents(corrected)] == ['2']
    assert 'Alpha decreases.' in answer_paragraph(corrected)
