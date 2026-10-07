from .config import STABILITY_THRESHOLD

def decide(evidence, source_test, query_test, classifier_available=True, query_relevant=True):
    support = [row for row in evidence if row['label'] == 'SUPPORT']
    contradiction = [row for row in evidence if row['label'] == 'CONTRADICT']
    groups = {row['source_group'] for row in support}
    components = {'support_max': max((row['support'] for row in evidence), default=0),
                  'contradiction_max': max((row['contradiction'] for row in evidence), default=0),
                  'distinct_support_groups': len(groups), 'ablation_survival': source_test['survival'],
                  'query_jaccard': query_test['mean_jaccard'], 'query_relevant': query_relevant}
    if not classifier_available:
        decision, status, reason = 'ABSTAIN', 'UNVERIFIED', 'Stance model unavailable; retrieval similarity alone cannot verify claims.'
    elif not query_relevant:
        decision, status, reason = 'ABSTAIN', 'INSUFFICIENT', 'Retrieved evidence has weak query relevance under the prototype gate.'
    elif contradiction and support:
        decision, status, reason = 'QUALIFY', 'CONTESTED', 'Retrieved passages both support and contradict this claim.'
    elif contradiction:
        decision, status, reason = 'REWRITE', 'CONTRADICTED', 'The claim is contradicted; replace it with an attributed retrieved passage.'
    elif not support:
        decision, status, reason = 'ABSTAIN', 'INSUFFICIENT', 'No retrieved passage reaches the support threshold.'
    elif len(groups) < 2 or source_test['survival'] != 1.0:
        decision, status, reason = 'QUALIFY', 'FRAGILE', 'Single-source dependence or failure to survive a source removal.'
    elif query_test['mean_jaccard'] < STABILITY_THRESHOLD:
        decision, status, reason = 'QUALIFY', 'UNSTABLE', 'Top-K retrieval changes under wording perturbations.'
    else:
        decision, status, reason = 'KEEP', 'ROBUST_WITHIN_CORPUS', 'Supported by distinct source groups and survives tested evidence perturbations.'
    return {'decision': decision, 'status': status, 'reason': reason, 'components': components}
