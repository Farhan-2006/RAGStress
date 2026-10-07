import numpy as np
from src.stance import Stance
from src.sparse_retriever import BM25
from src.hybrid_retriever import Retriever

def make_classifier(text):
    corpus = {'a': {'id': 'a', 'title': '', 'text': text}}
    classifier = Stance.__new__(Stance)
    classifier.retriever = Retriever(corpus, BM25(corpus))
    classifier.predict = lambda pairs: np.array([[.99, .005, .005] for _ in pairs])
    return classifier

def test_unrelated_gene_cannot_be_decisive_counterevidence():
    text = 'Deleting p53 changes apoptosis.'
    classifier = make_classifier(text)
    rows = classifier.classify('PPM1D suppresses p53.', [{'source_id': 'a', 'chunk_id': 'a:0', 'passage': text, 'rank': 1}])
    assert all(row['label'] == 'NEUTRAL' for row in rows)
    assert all('ppm1d' in row['missing_entity_anchors'] for row in rows)

def test_hyphenated_entity_anchor_is_recognized():
    text = 'Antiretroviral therapy prevents HIV-associated tuberculosis.'
    classifier = make_classifier(text)
    rows = classifier.classify(text, [{'source_id': 'a', 'chunk_id': 'a:0', 'passage': text, 'rank': 1}])
    assert rows[0]['missing_entity_anchors'] == []
