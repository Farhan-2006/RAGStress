import math
import pytest
from src.preprocessing import tokenize, chunks
from src.sparse_retriever import BM25, Tfidf
from src.hybrid_retriever import rrf, Retriever
from src.evaluation import metrics
from src.stability import jaccard
from src.trust import decide

@pytest.fixture
def corpus():
    return {'a': {'id': 'a', 'title': '', 'text': 'alpha alpha beta'},
            'b': {'id': 'b', 'title': '', 'text': 'alpha gamma'},
            'c': {'id': 'c', 'title': '', 'text': 'delta'}}

def test_normalization_preserves_negation():
    assert tokenize('NOT naïve p53 IL-6 １０') == ['not', 'na', 've', 'p53', 'il-6', '10']

def test_bm25_exact_score_and_exclusion(corpus):
    model = BM25(corpus)
    assert model.postings['alpha'] == {'a': 2, 'b': 1}
    hits = model.retrieve('alpha', 3)
    expected = math.log(1 + (3 - 2 + .5) / (2 + .5)) * 2 * 2.2 / (2 + 1.2 * (.25 + .75 * 3 / 2))
    assert hits[0]['source_id'] == 'a'
    assert hits[0]['score'] == pytest.approx(expected)
    assert [h['source_id'] for h in model.retrieve('alpha', 3, {'a'})] == ['b']
    assert model.retrieve('unknown') == []

def test_tfidf_cosine_and_no_overlap(corpus):
    model = Tfidf(corpus)
    assert model.retrieve('delta')[0]['score'] == pytest.approx(1.0)
    assert model.retrieve('unknown') == []

def test_rrf_unique_sources_and_math():
    hits = rrf([{'source_id': 'a', 'bm25': 2}, {'source_id': 'b', 'bm25': 1}],
               [{'source_id': 'b', 'dense': .9}, {'source_id': 'a', 'dense': .8}])
    assert len(hits) == 2
    assert hits[0]['score'] == pytest.approx(1 / 61 + 1 / 62)
    assert hits[0]['source_id'] == 'a'  # deterministic ID tie break

def test_metrics_duplicates_and_denominators():
    result = metrics(['a', 'a', 'z', 'b'], {'a': 1, 'b': 1}, 5)
    assert result['P@5'] == .4
    assert result['Recall@5'] == 1
    assert result['MRR@5'] == 1
    assert result['nDCG@5'] == pytest.approx((1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))

def test_chunks_keep_source_and_size():
    windows = chunks({'id': 'a', 'title': 'T', 'text': ' '.join(['word'] * 450)}, 150)
    assert len(windows) == 3
    assert all(w['source_id'] == 'a' and len(w['text'].split()) <= 150 for w in windows)

def test_jaccard_empty_is_not_stable():
    assert jaccard([], []) == 0
    assert jaccard(['a', 'b'], ['b', 'c']) == pytest.approx(1 / 3)

def test_question_rewrites_preserve_subject_effect_and_object():
    from src.stability import variants
    result = variants('Does antiretroviral therapy reduce tuberculosis incidence?')
    assert result[0] == 'Is tuberculosis incidence reduced by antiretroviral therapy?'
    assert all('antiretroviral therapy' in query and 'tuberculosis incidence' in query for query in result)

def test_decision_gates():
    row = {'label': 'SUPPORT', 'support': .99, 'contradiction': .01, 'source_group': 'g1'}
    query_test = {'mean_jaccard': .8}
    assert decide([row], query_test)['decision'] == 'KEEP'
    assert decide([row, dict(row, source_group='g2')], query_test)['decision'] == 'KEEP'
    assert decide([row], query_test, False)['decision'] == 'ABSTAIN'
    contradict = dict(row, label='CONTRADICT', support=.01, contradiction=.99)
    assert decide([contradict], query_test)['decision'] == 'REWRITE'
    assert decide([row, contradict], query_test)['status'] == 'CONTESTED'

def test_source_excluded_before_reranking(corpus):
    retriever = Retriever(corpus, BM25(corpus))
    assert all(h['source_id'] != 'a' for h in retriever.retrieve('alpha', 5, 'bm25', {'a'}))

def test_dense_missing_never_silently_becomes_sparse(corpus):
    with pytest.raises(RuntimeError):
        Retriever(corpus, BM25(corpus)).retrieve('alpha')
