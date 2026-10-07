# Experiment ledger

## Protocol frozen before held-out measurements
- Corpus: downloaded BEIR SciFact, original SciFact release for offline stance/rationale evaluation; SHA256 in data manifest.
- No fine-tuning, no test-dependent hyperparameter selection. BM25 k1=1.2/b=.75; RRF c=60/pool=50; output K=5; NLI=.80; stability=.40; dense query relevance=.30. All prototype gates.
- Retrieval evaluation: all 300 BEIR test queries; macro P@5, Recall@5, MRR@5, nDCG@5; source-level unique IDs.
- Novelty diagnostic: seeded random test-query subset (seed 358); record IDs. Wording probes are template wrappers, not validated paraphrases.
- Counter retrieval: original-dev contradictory gold sources. Union of 4 searches has larger budget than initial search; state this limitation.
- Gold ablation: remove highest ranked relevant source, evaluate only available other gold sources. Claims with zero annotated alternatives cannot yield alternative-recall estimates.
- Stance diagnosis: seeded original dev rationale pairs, thresholded NLI; gold rationale input isolates classifier and is not end-to-end accuracy.
- Human judgments remain pending until real team members complete worksheet.

## Engineering observations
- Bundled Python's temporary directory lacked write access; a workspace-local temp directory fixed environment creation.
- Sandbox networking blocked package DNS; authorized public dependency/data downloads used network escalation.
- Bundled Python certificate trust failed on dataset host; Windows trusted downloader recovered the real dataset. Added truststore support, never disabled TLS checks.
- All actual runtime/results and later failures should be recorded below; no claimed measurements until outputs exist.

## Completed actual measurements
- All 300 test queries: TF-IDF P@5=.1513 / Recall@5=.7011 / MRR@5=.5607 / nDCG@5=.5888; BM25 .1487/.6911/.5971/.6161; dense .1600/.7320/.5967/.6251; hybrid .1647/.7526/.6196/.6487. Full JSON contains unrounded values and uncertainty intervals.
- Dense indexing encoded 10,060 windows; approximately 9 minutes on this CPU. Reused the identical embeddings when pinning their originally downloaded model revision and migrating cache filename. No simulated embeddings.
- Hybrid Recall@5 and nDCG@5 bootstrap deltas are positive; MRR interval includes zero. Evaluation times reuse dense query cache for hybrid, so do not compare these times as unbiased latency.
- 50 seeded query probes: Jaccard BM25=.6741, dense=.7599, hybrid=.7190.
- 11 annotated-contradiction queries in that sample: BM25 K/expanded/original4K recall=.7879/.7879/.8788; dense=.8182/.9091/1.0; hybrid=.7879/.7879/.9091. Expanded counter queries do not beat the equal-slot-budget original query here.
- Gold alternative sources exist for only 5 BM25, 4 dense and 5 hybrid ablation cases; respective alternative recall=.30/1.0/.50. Small eligible sample: no broad source-independence claim.
- Initial MiniLM2 NLI dev diagnosis: 9/50 correct labels, 11/50 decisive (threshold .80). Labels/mapping sanity checks confirmed correct; failure is domain transfer, not inverted label order.
- Changed to DeBERTa-v3-small after this exploratory failure. Training diagnostic: 15/50 correct, 16/50 decisive. Dev diagnostic: 13/50 correct, 17/50 decisive (13/17 selective accuracy). No threshold optimized on dev; model selection/error analysis is exploratory and must be disclosed.
- Live error: confidently mislabeled unrelated p53 evidence for PPM1D. Added named-entity presence gate, then an IDF-weighted topical-coverage gate >=.25 after another unrelated neural-development contradiction. These are engineering safeguards, not statistically calibrated truth guarantees. They may reject valid aliases/synonyms.
- NLI evaluates a full retrieved passage as well as matched sentences to preserve context. Isolated rationale diagnosis bypasses runtime retrieval/gates and remains explicitly a classifier-only measurement.
- Streamlit verification initially stalled inside the Windows sandbox; a minimal UI test succeeded outside it. Actual AppTest then revealed model-cache import ordering in the test harness; fixed by configuring HF_HOME before model imports. Live app test succeeded with real corpus retrieval and local LLM.
- Core tests: 15 pass before the final topical gate verification. Final validation status saved separately in results/verification.json.
- Final functional suite: 16 tests passed. Real Streamlit hybrid interaction passed with local LLM, actual Top-5 scores and all stress-test panels.
- A complete local-LLM question succeeds: "What potential does antiretroviral therapy have to prevent HIV-associated tuberculosis?" -> QUALIFY, supported source 4883040, fragile source ablation. Export is demo-grounded-question.json; approximately 29 seconds before length-grouped NLI batching optimization.
- Remaining failure: some neural-development passages are falsely treated as contradicting a transplantation claim even after topical gating. This is disclosed as a model limitation, not presented as correct verification. Ordinary question generation can also fail the strict audit despite a broadly plausible answer.

## Abstention correction (6 October 2026)
The generated paraphrase for the tuberculosis-incidence question failed the support gate even though ranked sources contained relevant findings. Added a question-only recovery path that audits an attributed exact corpus quotation, preserving the failed original audit. The actual local-model run now returns REWRITE with source 4883040 and FRAGILE evidence; see results/fix-answerable.json. This is correction, not validation of the rejected wording. The running Streamlit server was restarted to replace stale imports. Functional suite: 19 tests passed, including recovery, out-of-corpus rejection and direct-claim preservation. Recovery relevance gates are exploratory, not calibrated truth probabilities.

## Claims we do NOT make
- No superiority of counter-query expansion in these measurements.
- No end-to-end unsupported-answer reduction number without completed human review.
- No validated semantic-paraphrase robustness from template probes alone.
- No scientific independence merely from different source IDs.
- No calibrated probability of correctness from NLI or retrieval scores.
