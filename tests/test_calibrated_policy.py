from src.trust import decide
from src.stance import Stance
from src.sparse_retriever import BM25
from src.hybrid_retriever import Retriever
import numpy as np

def test_single_strong_source_and_moderate_instability_can_keep():
    row={'label':'SUPPORT','support':.95,'contradiction':.02,'source_id':'1'}
    assert decide([row],{'mean_jaccard':.30})['decision']=='KEEP'
    assert decide([row],{'mean_jaccard':.05})['decision']=='QUALIFY'
    assert decide([dict(row,label='WEAK_SUPPORT',support=.65)],{'mean_jaccard':.8})['decision']=='QUALIFY'

def test_conflict_and_unavailable_verification_still_matters():
    support={'label':'SUPPORT','support':.95,'contradiction':.01,'source_id':'1'}
    counter=dict(support,label='CONTRADICT',support=.01,contradiction=.99)
    assert decide([support,counter],{'mean_jaccard':.8})['status']=='CONTESTED'
    assert decide([counter],{'mean_jaccard':.8})['decision']=='REWRITE'
    assert decide([],{'mean_jaccard':.8})['decision']=='ABSTAIN'
    assert decide([support],{'mean_jaccard':.8},False)['decision']=='ABSTAIN'

def classifier(text,score):
    corpus={'1':{'id':'1','title':'','text':text}}
    model=Stance.__new__(Stance)
    model.retriever=Retriever(corpus,BM25(corpus))
    model.predict=lambda pairs:np.array([score]*len(pairs))
    return model,[{'source_id':'1','chunk_id':'1:0','passage':text,'rank':1}]

def test_gene_mismatch_blocks_even_strong_semantic_score():
    model,hits=classifier('p53 affects cancer progression.',[.01,.98,.01])
    rows=model.classify('PPM1D affects cancer progression.',hits)
    assert all(r['label']=='NEUTRAL' and r['hard_entity_mismatch'] for r in rows)

def test_named_alias_is_allowed():
    model,hits=classifier('p53 affects cancer progression.',[.01,.98,.01])
    rows=model.classify('TP53 affects cancer progression.',hits)
    assert all(r['label']=='SUPPORT' and not r['missing_entity_anchors'] for r in rows)
