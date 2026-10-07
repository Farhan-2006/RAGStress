"""Pipeline contract tests use test doubles only; real demos use corpus/model outputs."""
from src.pipeline import Pipeline
from src.sparse_retriever import BM25
from src.hybrid_retriever import Retriever

class FixtureGenerator:
    def generate(self, query, hits):
        return 'Alpha is increased. [1]', 'test-double', None

def test_missing_classifier_abstains_and_keeps_trace():
    corpus = {'1': {'id': '1', 'title': 'Study', 'text': 'Alpha is increased.'}}
    result = Pipeline(Retriever(corpus, BM25(corpus)), FixtureGenerator()).run('alpha', 'bm25')
    assert result['decision'] == 'ABSTAIN'
    assert result['claims'][0]['status'] == 'UNVERIFIED'
    assert result['retrieved'][0]['source_id'] == '1'
    assert result['claims'][0]['counter_evidence']['runs']
    assert '[1]' in result['initial_answer']

def test_unaudited_sentences_are_never_kept():
    class TooLong:
        def generate(self, query, hits):
            return 'Alpha grows. Beta grows. Gamma grows. Delta grows.', 'test-double', None
    corpus = {'1': {'id': '1', 'title': '', 'text': 'Alpha beta gamma delta.'}}
    result = Pipeline(Retriever(corpus, BM25(corpus)), TooLong()).run('alpha', 'bm25')
    assert len(result['claims']) == 3
    assert result['unexamined_sentences_omitted']
    assert 'Delta grows' not in result['final_answer']
