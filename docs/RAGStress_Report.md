# RAGStress

Evidence Auditing and Counterfactual Retrieval Stress Testing
CSD358 Information Retrieval Hackathon | T1: RAG and Trustworthy Answers

| Team member | Roll number |
| --- | --- |
| Farhan Naik | 2410110422 |
| Hunar Bhatia | 2410110427 |
| Aniket Sharma | 2410110516 |

## Abstract

RAGStress asks whether a generated answer remains defensible when retrieval is challenged. It combines an inspectable BM25 inverted index with dense sentence-window retrieval and Reciprocal Rank Fusion (RRF). A separate natural language inference (NLI) model audits claims, searches the corpus for counter-evidence, and measures retrieval stability under query variants. Evidence gates produce KEEP, QUALIFY, REWRITE or ABSTAIN. On 300 BEIR SciFact queries, hybrid nDCG@5 was 0.6487 versus 0.6161 for BM25. In a separate 50-case historical answer-auditing experiment, model-assessed groundedness rose from 65.67% to 87.10%, while coverage fell from 100% to 96% and mean latency increased from 1.08 to 11.83 seconds. Verification-only matched the full system on aggregate quality. These findings support hybrid retrieval and evidence auditing, but do not establish an additional quality benefit from stress testing or independently verified answer accuracy.

Version disclosure: the 50-case experiment evaluated the earlier short-answer revision, preserved in RAGStress_Evaluated_Snapshot.zip. The current interface displays one paragraph and approved documents separately; that revised answer format has not received a new 50-case benchmark. Historical metrics must not be attributed to it.

## 1. Problem, users and track relevance

A fluent answer can misrepresent a retrieved abstract, ignore contradictory findings, or appear reliable only because of query wording. Students and researchers need both an answer and an inspectable explanation of its evidential limits. This directly belongs to T1: retrieval supplies every evidence item, and answer generation is followed by evidence-based qualification, correction or abstention. The system operates within a scientific abstract corpus; it is not a medical advice service or a search of all scientific literature.

SciFact studies scientific claim verification using abstracts and evidence rationales [1]. BEIR supplies a reproducible ranked-retrieval benchmark [2]. RRF provides rank-based sparse/dense fusion [3]. FActScore motivates fine-grained factual assessment [4], although our sentence/semicolon candidates are a simpler approximation to atomic claims. These works inform the project; no existing project implementation was copied.

## 2. How Information Retrieval is used

The corpus contains 5,183 SciFact abstracts. An abstract is the source document; overlapping sentence windows are passages. Preprocessing produces 10,060 windows, bounded to approximately 150 words with one-sentence overlap. Source-level ranking prevents several windows from one paper receiving several fusion votes.

| Principle / location | Implementation and purpose |
| --- | --- |
| Documents and normalization
src/data.py; preprocessing.py | Unicode NFKC, case folding and regex tokenization. Negation and biomedical terms are retained; no stemming or blanket stop-word removal. |
| Inverted index, TF, DF, IDF
src/sparse_retriever.py | Explicit term -> document -> term-frequency postings. Document frequency and IDF can be inspected to explain sparse matches. |
| BM25 and heap Top-K
src/sparse_retriever.py | TF saturation and document-length normalization; k1=1.2, b=0.75. Heap selection returns a real ranked list. |
| TF-IDF and cosine baseline
src/sparse_retriever.py | Sublinear TF and L2-normalized vectors; dot product equals cosine. Provides a simpler comparison. |
| Dense semantic retrieval
src/dense_retriever.py | all-MiniLM-L6-v2 [5] normalized window embeddings; cosine scores; maximum window score per source. Embeddings are cached. |
| Hybrid rank fusion
src/hybrid_retriever.py | RRF pools 50 results per retriever, constant 60, default K=5. Scores and source IDs remain visible. |
| Evaluation and inspection
src/evaluation.py; app.py | Ranked source IDs are matched to public qrels. UI exposes BM25, dense and fusion scores, and an index dictionary. |

BM25 IDF(t) = ln[1 + (N - df(t) + 0.5)/(df(t) + 0.5)]. A term contributes IDF(t) * tf(t,d)*(k1+1) / [tf(t,d) + k1*(1-b+b*|d|/avgdl)]. Repeated words saturate; long documents are normalized. Query-term inspection makes the scoring interpretable.

For source d, RRF(d) = sum over retrievers of 1/(60 + rank(d)); an absent source contributes zero. RRF combines rankings without pretending BM25 and cosine scores share a scale. The dense score is cosine(query, window), with the best window representing each source. Neither the generator nor the verifier chooses the ranked documents.

Optional techniques were kept out when they did not add a clear benefit. There is no unnecessary agent framework, crawling dependency, phrase index or frontend animation layer. The working retrieval implementation and quantitative baseline are the central course contributions.

## Pipeline diagram

Question -> normalization -> BM25 + dense windows -> RRF Top-K -> grounded draft -> claim candidates / separate NLI -> corpus counter-search + query stability -> evidence gates -> final paragraph and approved documents.

## 3. Beyond IR: generation, verification and decisions

Local FLAN-T5-small [7] generates a draft from retrieved passages. The current prompt requests three explanatory sentences; independently audited retrieved findings may expand an overly short draft. Claim extraction uses sentence and semicolon boundaries with a cap of six candidates. The separate DeBERTa NLI model [6] labels retrieved evidence approximately SUPPORT, CONTRADICT or NEUTRAL. It is not the generator evaluating itself, but it remains a fallible generic-domain model.

| Gate | Prototype rule and interpretation |
| --- | --- |
| Support | Strong aligned entailment >=0.80; moderate >=0.60. Entity/topic checks combine lexical IDF overlap and semantic alignment. Specific mismatched numbered entities block decisive labels. |
| Contradiction | NLI >=0.90 plus confirmation using displayed context and adequate entity/topic alignment. A negative search query alone is not evidence of contradiction. |
| Stability | Three wording variants; unique-source Top-K Jaccard. Mean overlap below 0.15 qualifies the answer. Moderate variation does not automatically reject strong support. |
| Final action | KEEP for strong support; QUALIFY for moderate support, conflict or severe instability; REWRITE for audited correction; ABSTAIN when aligned evidence is insufficient. |

These are calibrated prototype gates, not a probability of truth or scientifically established universal thresholds. Individual support, conflict and stability signals remain visible. One strong relevant source can suffice. Source ablation was removed from the final implementation and is not claimed as a working feature. The answer is one paragraph; approved documents appear in a separate section only when used for retained content or a correction. Retrieval alone does not approve a source.

## 4. Novelty and creativity

Hybrid retrieval and citation display are baseline capabilities, not the claimed novelty. The project contribution is an integrated evidence-auditing workflow that actively retrieves possible counter-evidence, changes query wording, assesses individual claims and exposes reasons to qualify, rewrite or abstain. This is a novel combination and implementation for this project, not a claim that RAG robustness or abstention has never been studied. The current stress tests cover counter-search and query perturbation; source-removal robustness remains outside the final scope.

## 5. Evaluation: protocol and ranked retrieval

Experiment 1 uses all 300 BEIR SciFact test queries and public document relevance judgments. Metrics are macro-averaged over queries at K=5 with unique source IDs: Precision@5 = relevant returned / 5; Recall@5 = relevant returned / all judged relevant; MRR@5 uses the reciprocal rank of the first relevant result; nDCG@5 discounts relevant documents by rank and normalizes against the ideal ranking. Public gold judgments are the justified alternative to inventing manual judgments. Binary qrels describe document relevance, not full answer correctness.

| Method | P@5 | Recall@5 | MRR@5 | nDCG@5 |
| --- | --- | --- | --- | --- |
| TF-IDF | 0.1513 | 0.7011 | 0.5607 | 0.5888 |
| BM25 | 0.1487 | 0.6911 | 0.5971 | 0.6161 |
| Dense | 0.1600 | 0.7320 | 0.5967 | 0.6251 |
| Hybrid RRF | 0.1647 | 0.7526 | 0.6196 | 0.6487 |

Hybrid improves nDCG@5 by 0.0326 over BM25. A paired bootstrap with 2,000 resamples and seed 358 gives a 95% interval of [0.0063, 0.0595]. Recall gain is 0.0614, interval [0.0272, 0.0969]. The MRR gain interval includes zero, so that gain is not established. Retrieval timing is not used to claim a speed advantage because cached query embeddings affect comparability.

Artifacts: results/retrieval-test.csv, retrieval-test.json and per-query records. The retrieval module remains unchanged by the paragraph presentation update; the stored values remain measured results from the recorded run.

## 5. Evaluation: historical answer-auditing comparison

Experiment 2 uses 50 original SciFact development examples, balanced at 25 SUPPORT and 25 CONTRADICT, selected with seed 358. Available stance annotations make this suitable for diagnostics; earlier prototype exposure means it is exploratory, not a pristine unseen test. Calibration used 80 separate original-training rationale pairs. Gold labels never enter runtime generation or evidence decisions. Vanilla, verification-only and full auditing share identical initial hybrid retrieval, K=5 and generator drafts; 150 timed executions rotate system order and exclude initial model loading.

| Measure | Vanilla | Verification only | Full audit |
| --- | --- | --- | --- |
| NLI-supported final claims | 49/67 (73.13%) | 60/62 (96.77%) | 60/62 (96.77%) |
| Grounded: support without contradiction | 44/67 (65.67%) | 54/62 (87.10%) | 54/62 (87.10%) |
| Unsupported final claims | 8/67 (11.94%) | 2/62 (3.23%) | 2/62 (3.23%) |
| Fully supported answers (NLI) | 27/50 (54%) | 40/50 (80%) | 40/50 (80%) |
| Answer coverage | 100% | 96% | 96% |
| Gold rationale sentence recall | 56.67% | 56.67% | 56.67% |
| Automated answer-stance agreement | 21/50 (42%) | 26/50 (52%) | 26/50 (52%) |
| Mean latency, seconds | 1.083 | 8.983 | 11.834 |
| Median latency, seconds | 0.832 | 6.806 | 9.396 |

Claim rates use final factual candidates, so denominators differ after rewriting. Groundedness requires model-assessed support without model-assessed contradiction; support and contradiction rates can overlap. These are automated evidence diagnostics, not independently judged factual accuracy. The NLI judge is reused from the auditor, and source-quote recovery naturally favors entailment. Gold free-form answer accuracy, answer precision/recall/F1, selective answer accuracy and evidence precision are not available because annotations do not exhaustively label newly generated statements.

The common initial retrieval is P@5=0.172, Recall@5=0.820, MRR@5=0.6817 and nDCG@5=0.7134. Identical retrieval is expected in this controlled auditor comparison. It must not be interpreted as the auditor improving initial IR rankings. These results belong to the frozen short-answer snapshot, not the current one-paragraph output.

## 5. Evaluation: stress tests, errors and interpretation

| Observed diagnostic | Historical full-system result |
| --- | --- |
| Final actions | 26 KEEP; 6 QUALIFY; 16 REWRITE; 2 ABSTAIN |
| Counter-evidence flagged by NLI | 17/64 audited claims (26.56%) |
| Unqualified drafts challenged | 16/50 (32%); operational hedge-word diagnostic |
| Query stability | Mean original-to-variant Jaccard 0.7319 |
| Severe instability | 2/300 variant-pair overlaps <0.15; 0/50 query means <0.15 |
| Gold refuting-source detection | 10/25 (40%) before and after counter-search |
| Full audit vs verification-only | Same aggregate quality and decisions; 2.85 seconds higher mean latency |

No additional gold-refuting document was discovered by counter-search in this sample, and no query had severe mean instability. Therefore stress tests provide observable diagnostics, but their incremental quality benefit is not demonstrated here. Verification-only is the stronger efficiency result. The full audit costs about 10.93 times vanilla mean latency; paired bootstrap estimates the extra cost at 10.75 seconds, interval [9.07, 12.71].

Fully supported answer incidence increases by 26 percentage points, bootstrap interval [14, 38] points; McNemar p=0.000244. Automated answer-stance agreement increases by 10 points, but McNemar p=0.3323 does not establish significance. Neither test removes the limitations of a reused neural judge or establishes independent answer correctness.

| Real evaluated claim | Observed limitation or improvement |
| --- | --- |
| 674: LDL cholesterol has no involvement in cardiovascular disease. | Gold CONTRADICT: vanilla stance mismatch; full audit REWRITE aligns with the gold stance. |
| 217: CX3CR1 on Th2 cells promotes T-cell survival. | Gold SUPPORT: automated answer stance remained unresolved despite a KEEP decision. KEEP is not proof of truth. |
| 233: Cell-autonomous sex determination does not occur in Galliformes. | Gold CONTRADICT: vanilla was judged fully supported, while full auditing returned QUALIFY. Model diagnostics can disagree. |

The confusion matrices expose an asymmetry: gold-CONTRADICT alignment improves from 6/25 to 17/25, while gold-SUPPORT alignment falls from 15/25 to 9/25. This guards against presenting only a favorable overall rate. A larger independently judged question set, real mixed-evidence examples and deliberately unanswerable queries are needed to evaluate stress testing and abstention convincingly.

Recorded evidence: evaluation/results/raw_results.json, summary.json, comparison.csv, statistics.json, stress_metrics.json, confusion_matrices.json and error_analysis.json. The 50-case subset and protocol are stored in evaluation/evaluation_dataset.json and the frozen manifests. Current paragraph smoke checks include a real ART question producing KEEP and an out-of-corpus fantasy question producing ABSTAIN; they are functional checks, not a replacement benchmark.

## 6. Limitations and next steps

Scientific abstracts omit full study methods and can reflect conflicting settings. Public qrels are incomplete for novel counter-searches. MiniLM, FLAN-T5-small and generic NLI have domain and context-window limitations. Sentence/semicolon splitting can merge facts or miss qualifiers. Controlled query variants are not human-certified paraphrases; low overlap measures wording sensitivity, not falsehood. Fixed gates are prototype choices, and a relevant single source does not demonstrate independent scientific consensus. Source ablation is absent. The current answer-paragraph revision still needs a fresh complete benchmark.

Immediate roadmap: freeze the current implementation; rerun all paired evaluations; obtain blinded human judgments of claims, stance and source relevance; add mixed-evidence and unanswerable cases; inspect SUPPORT-class regressions; and measure stress-test-only gains. A later course project can introduce better atomic claim extraction, domain-specific verification, carefully validated paraphrases, and optional independent-source ablation. Prioritize justified quality gains and reproducibility over a larger UI.

## 7. Work division

Proposed balanced ownership below must be confirmed against actual contributions before submission. It is a planning allocation, not a statement that these historical tasks were performed by these people. No percentage split is required by the assignment.

| Member / roll number | Proposed ownership to confirm |
| --- | --- |
| Farhan Naik
2410110422 | Dataset and preprocessing; sparse/dense retrieval and RRF; reproducible setup; demonstrates retrieval and scoring. |
| Hunar Bhatia
2410110427 | Grounded generation, claim and NLI checks; counter-evidence, stability and trust decisions; demonstrates evidence auditing. |
| Aniket Sharma
2410110516 | Evaluation, statistics and graphs; Streamlit source inspection; documentation and video coordination; demonstrates results and limitations. |

## AI-use declaration and reproducibility

OpenAI Codex assisted with original implementation, debugging, UI revisions, calibration experiments, evaluation tooling and report/script drafting. FLAN-T5-small generates drafts, MiniLM encodes passages, and DeBERTa performs NLI during operation. Public libraries include NumPy, scikit-learn, Sentence Transformers, Transformers, PyTorch, Streamlit and pytest; versions and model revisions are recorded in the repository. Dataset, model and paper credits are provided below. No existing project was cloned, no benchmark numbers or human judgments were fabricated, and the team must review the code, results and final ownership declaration.

Reproduce from README: create the Python environment, install requirements, run python -m src.cli download and python -m src.cli warmup, then python -m streamlit run app.py. Run python -m pytest -q for core checks. Run python run_evaluation.py --no-resume to evaluate the current revision; preserve historical outputs first. Use the evaluated snapshot to rebuild the historical report. Local default inference requires no API key. The current core suite passed 40 tests. Submit the reviewed GitHub repository, this report PDF, and a 5-8 minute unlisted live-demo video without slides; every member must speak.

## References (excluded from the main-page limit)

0. CSD358 IR Mid-term Assignment 2026. Official assignment PDF supplied by the course; primary requirements and rubric source.

1. Wadden et al. (2020). Fact or Fiction: Verifying Scientific Claims. EMNLP.

https://arxiv.org/abs/2004.14974

2. Thakur et al. (2021). BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models.

https://arxiv.org/abs/2104.08663

3. Cormack, Clarke and Buettcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods. SIGIR.

https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/

4. Min et al. (2023). FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.

https://arxiv.org/abs/2305.14251

5. Sentence Transformers. all-MiniLM-L6-v2 model card.

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

6. Cross-Encoder. nli-deberta-v3-small model card; trained on general-domain NLI data.

https://huggingface.co/cross-encoder/nli-deberta-v3-small

7. Google. FLAN-T5-small model card.

https://huggingface.co/google/flan-t5-small

8. Allen Institute for AI. SciFact data schema and original annotations.

https://github.com/allenai/scifact/blob/master/doc/data.md

9. Project repository (check access and publish the current revision before submission).

https://github.com/Farhan-2006/RAGStress

Primary measurement files are included with the submission source and evaluated snapshot. Current code and historical results are identified separately throughout this report. The final team contribution section and video ownership statements require team confirmation.
