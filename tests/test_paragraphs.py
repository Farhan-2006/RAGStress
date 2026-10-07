from src.pipeline import Pipeline
from src.hybrid_retriever import Retriever
from src.sparse_retriever import BM25
from src.claims import extract_claims

FINDING = 'Alpha treatment reduces beta disease incidence in adults significantly.'
DETAIL = 'Alpha treatment reduced beta disease incidence by 40 percent in adults in the study.'


class Generator:
    def generate(self, query, hits):
        return FINDING, 'fixture', None


class Classifier:
    def __init__(self, reject_detail=False):
        self.checked = []
        self.reject_detail = reject_detail

    def classify(self, claim, hits):
        self.checked.append(claim)
        supported = claim == FINDING or (claim == DETAIL and not self.reject_detail)
        return [{'source_id':h['source_id'], 'quote':DETAIL, 'label':'SUPPORT' if supported else 'NEUTRAL',
                 'support':.99 if supported else .01, 'contradiction':.01} for h in hits]


def run(reject_detail=False):
    corpus = {'1':{'id':'1', 'title':'Alpha treatment findings', 'text':FINDING + ' ' + DETAIL}}
    classifier = Classifier(reject_detail)
    result = Pipeline(Retriever(corpus, BM25(corpus)), Generator(), classifier).run(
        'alpha treatment beta disease incidence adults', 'bm25')
    return result, classifier


def test_source_detail_is_audited_before_appearing_in_the_paragraph():
    result, classifier = run()
    assert DETAIL in classifier.checked
    assert DETAIL in result['final_answer']
    assert '\n' not in result['final_answer']
    assert any(c.get('origin') == 'retrieved_context' for c in result['claims'])


def test_unverified_detail_cannot_pad_the_answer():
    result, classifier = run(True)
    assert DETAIL in classifier.checked
    assert DETAIL not in result['final_answer']
    assert result['answer_context'] == []


def test_paragraph_claims_are_not_silently_cut_after_three_sentences():
    claims, truncated = extract_claims('Alpha increases. Beta decreases. Gamma varies. Delta stabilizes.')
    assert len(claims) == 4 and not truncated
