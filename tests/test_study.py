import pytest
from src.evaluation import metrics
from evaluation.analysis import classification, paired_stats
from evaluation.study import gold_recall, answer_stance, assert_frozen


def test_rank_metrics_account_for_missing_and_duplicate_results():
    result = metrics(['wrong', 'a', 'a'], {'a': 1, 'b': 1}, 5)
    assert result['P@5'] == .2
    assert result['Recall@5'] == .5
    assert result['F1@5'] == pytest.approx(2 / 7)
    assert result['MRR@5'] == .5
    assert metrics([], {}, 5)['nDCG@5'] == 0


def test_gold_sentence_recall_requires_matching_source_and_counts_unique_sentences():
    example = {'gold_evidence': [
        {'doc_id': 'a', 'sentence_index': 0, 'text': 'The finding.'},
        {'doc_id': 'a', 'sentence_index': 0, 'text': 'The finding.'},
        {'doc_id': 'a', 'sentence_index': 1, 'text': 'A second finding.'}]}
    assert gold_recall([{'source_id': 'b', 'passage': 'The finding.'}], example) == 0
    assert gold_recall([{'source_id': 'a', 'passage': 'THE   finding.'}], example) == .5


def record(gold, predicted, supported, seconds):
    evaluation = {'answer_stance_diagnostic': {'predicted': predicted},
                  'gold_stance_agreement': gold == predicted,
                  'fully_supported_answer': supported}
    return {'example': {'gold_label': gold}, 'evaluation': evaluation, 'elapsed_seconds': seconds}


def test_abstention_and_unresolved_predictions_remain_recall_errors():
    examples = [record('SUPPORT', 'SUPPORT', True, 1),
                record('SUPPORT', 'ABSTAIN', False, 1),
                record('CONTRADICT', 'UNRESOLVED', False, 1),
                record('CONTRADICT', 'SUPPORT', False, 1)]
    rows = [{'example': r['example'], 'systems': {'full': r}} for r in examples]
    diagnostic = classification(rows, 'full')
    assert diagnostic['matrix'] == [[1, 0, 0, 1], [1, 0, 1, 0]]
    assert diagnostic['precision'] == .25
    assert diagnostic['recall'] == .25
    assert answer_stance('claim', [], None)['predicted'] == 'ABSTAIN'


def test_identical_paired_outcomes_have_no_evidence_of_difference():
    run = record('SUPPORT', 'SUPPORT', True, 1)
    rows = [{'systems': {'vanilla': run, 'ragstress': run}} for _ in range(3)]
    stats = paired_stats(rows)
    assert all(s['p_value'] == 1 for s in stats[:2])
    assert all(s['ci95'] == [0, 0] for s in stats[2:])


def test_study_runtime_still_matches_frozen_manifest():
    assert assert_frozen()['files']


def test_revision_mismatch_does_not_overwrite_previous_protocol(tmp_path, monkeypatch):
    import json
    import evaluation.study as study
    monkeypatch.setattr(study, 'OUT', tmp_path)
    monkeypatch.setattr(study, 'dataset', lambda limit: [{'id':'1'}])
    (tmp_path/'raw').mkdir()
    (tmp_path/'raw/000-1.json').write_text(json.dumps({'signature':'previous-revision'}))
    (tmp_path/'protocol.json').write_text('preserved historical protocol')
    with pytest.raises(RuntimeError, match='earlier answering version'):
        study.run_study(None, None, None, 1)
    assert (tmp_path/'protocol.json').read_text() == 'preserved historical protocol'
