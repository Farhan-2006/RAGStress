# Calibration and decision changes

The existing corpus, chunking, sparse/dense retrieval, RRF, local generator and separate DeBERTa verifier remain in place. The former extra source-dependency test was deleted from the active pipeline, interface, exports, novelty metrics and documentation. Exactly two stress tests remain: counter-evidence retrieval and query wording stability.

| Gate | Previous | Current | Reason |
|---|---|---|---|
| Strong support | .80 | .80 | Preserve a conservative threshold for unqualified retention. |
| Moderate support | None | .60 | Training-pair threshold grid supports a qualified intermediate outcome instead of discarding every sub-.80 score. |
| Contradiction | .80 | .90 plus displayed-passage context confirmation | A strong contradiction must survive context; generic NLI can mistake sentence fragments for refutation. |
| Weighted claim-term coverage | .25 mandatory over the passage | .15 over the candidate quotation OR semantic cosine .40 | Treat lexical alignment as a secondary route; scientific synonyms need not match surface words. |
| Missing entity | Any missing acronym blocks all decisive labels | Canonical aliases; family matching; strong semantic support may resolve generic acronyms | Avoid rejecting TP53/p53 and expanded names while retaining hard blocking for mismatched numbered entities. |
| Wording stability | Mean Jaccard <.40 qualifies | Only <.15 qualifies otherwise strong support | Moderate source-set variation is not enough to infer an unreliable answer. |
| Query relevance | Dense cosine .30 | .30 | Preserve protection against entirely unrelated questions. |
| Supporting-source count | Multiple groups and full survival required | One relevant strong supporting document is sufficient | SciFact often annotates one genuinely relevant study. |

Old policy: missing strong support meant withholding; one source or incomplete survival meant qualification; overlap below .40 also qualified. Current policy: strong aligned support can KEEP; moderate support QUALIFY; confirmed support/conflict QUALIFY; contradiction alone REWRITE; no support/unrelated evidence/unavailable verifier ABSTAIN. Severe wording instability qualifies strong support. The existing attributed-quotation recovery path remains separately reported.

These are prototype heuristics, not established scientific constants or calibrated truth probabilities. The threshold search used 80 seeded original-training gold rationale pairs (40 SUPPORT, 40 CONTRADICT), not the later comparison subset. The .60/.90 pair maximized training macro F1 under the fixed alignment rule, with conservative tie breaking. Context/entity safeguards are additional engineering constraints, so the threshold-only result is not the production result.

Old classification: 20 correct labels/80 pairs, 22 decisive predictions (27.5% coverage; 90.91% accuracy conditional on decisive labels). Threshold-only grid winner: 25 correct/80, 27 decisive predictions. Production safeguards yield 23 correct/80, 25 decisive predictions (31.25% coverage; 92% conditional accuracy). Production predictions and traces are in calibration/old_new_training_diagnosis.json; real end-to-end sanity cases are in calibration/sanity_runs.json. This calibration subset has no neutral examples and cannot establish general answer correctness.

An actually recorded antiretroviral-therapy question previously returned QUALIFY and now returns KEEP. Both answers attribute corpus evidence. The old executed record and new pre-freeze sanity record are preserved in calibration/previous_overrejection_examples.json, together with five gold-SUPPORT training examples previously labelled NEUTRAL. These are evidence-label diagnostics, not invented full-answer runs.

Runtime hashes were frozen before constructing the 50-case dev comparison. No thresholds were changed after inspecting its outcomes. Presentation and evaluation reporting can change without changing the frozen answering policy. Earlier prototype exposure to some dev examples makes this an exploratory comparison rather than a pristine unseen benchmark.
