from .config import STABILITY_THRESHOLD

def decide(evidence, query_test, classifier_available=True, query_relevant=True):
    strong = [r for r in evidence if r['label'] == 'SUPPORT']
    weak = [r for r in evidence if r['label'] == 'WEAK_SUPPORT']
    contradiction = [r for r in evidence if r['label'] == 'CONTRADICT']
    components = {'support_max': max((r['support'] for r in evidence), default=0),
                  'contradiction_max': max((r['contradiction'] for r in evidence), default=0),
                  'supporting_documents': len({r.get('source_id') for r in strong + weak}),
                  'strong_support': bool(strong), 'weak_support': bool(weak),
                  'query_jaccard': query_test['mean_jaccard'], 'query_relevant': query_relevant}
    if not classifier_available:
        decision, status, reason = 'ABSTAIN', 'UNVERIFIED', 'Evidence-checking model unavailable; similarity alone cannot verify claims.'
    elif not query_relevant:
        decision, status, reason = 'ABSTAIN', 'INSUFFICIENT', 'Retrieved evidence is unrelated under the prototype query-relevance gate.'
    elif contradiction and (strong or weak):
        decision, status, reason = 'QUALIFY', 'CONTESTED', 'Relevant passages support and contradict the claim.'
    elif contradiction:
        decision, status, reason = 'REWRITE', 'CONTRADICTED', 'Relevant corpus evidence contradicts the wording; replace it with an attributed passage.'
    elif strong:
        if query_test.get('tested', True) and query_test['mean_jaccard'] < STABILITY_THRESHOLD:
            decision, status, reason = 'QUALIFY', 'UNSTABLE', 'Strong support exists, but retrieval is severely sensitive to wording.'
        else:
            decision, status, reason = 'KEEP', 'SUPPORTED', 'Strong relevant support, no confirmed contradiction, and no severe wording instability.'
    elif weak:
        decision, status, reason = 'QUALIFY', 'WEAKLY_SUPPORTED', 'Moderate semantic support exists with topical alignment; wording should remain qualified.'
    else:
        decision, status, reason = 'ABSTAIN', 'INSUFFICIENT', 'No sufficiently aligned supporting or contradictory evidence could verify this claim.'
    return {'decision':decision,'status':status,'reason':reason,'components':components}
