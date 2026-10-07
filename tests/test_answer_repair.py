from src.answer_repair import repair_candidates
from src.pipeline import Pipeline
from src.sparse_retriever import BM25
from src.hybrid_retriever import Retriever

QUOTE = 'Alpha treatment reduces beta disease incidence in adults significantly.'

def setup():
    corpus = {'1': {'id': '1', 'title': 'Clinical findings', 'text': QUOTE}}
    return Retriever(corpus, BM25(corpus))

class BadGenerator:
    def generate(self, query, hits):
        return 'Alpha treatment cures every disease.', 'test-double', None

class QuoteOnlyClassifier:
    def classify(self, claim, hits):
        return [{'source_id': hit['source_id'], 'source_group': hit['source_id'],
                 'quote': QUOTE, 'label': 'SUPPORT' if claim == QUOTE else 'NEUTRAL',
                 'support': .99 if claim == QUOTE else .01, 'contradiction': .01}
                for hit in hits]

def test_failed_generation_replaced_by_audited_corpus_quote():
    result = Pipeline(setup(), BadGenerator(), QuoteOnlyClassifier()).run('alpha treatment beta disease incidence adults', 'bm25')
    assert result['decision'] == 'REWRITE'
    assert result['answer_repair']['quote'] == QUOTE
    assert result['superseded_claims'][0]['decision'] == 'ABSTAIN'
    assert 'cures every disease' not in result['final_answer']
    assert result['claims'][0]['decision'] == 'QUALIFY'
    assert result['claims'][0]['ablation']['survival'] == 0

def test_out_of_corpus_words_block_quote_recovery():
    retriever = setup()
    hits = retriever.retrieve('adults', 5, 'bm25')
    assert repair_candidates(retriever, 'certified dragon wingspan on Zorblax with adults', hits) == []

def test_claim_mode_never_substitutes_the_claim_with_a_quote():
    result = Pipeline(setup(), BadGenerator(), QuoteOnlyClassifier()).run('alpha treatment cures every disease', 'bm25', claim_mode=True)
    assert result['decision'] == 'ABSTAIN'
    assert result['answer_repair'] is None
