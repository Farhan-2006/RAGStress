# RAGStress / TrustRAG

Counterfactual Evidence Stress Testing for Retrieval-Augmented Generation

CSD358 - T1 | Hackathon prototype | 6 October 2026

DRAFT: team names, actual work division and human-judgment review remain to be completed. Results below come from actual runs, not proposed targets.

## 1. Problem and track relevance

Can a RAG system determine when to trust, qualify or abstain from its answer by actively stress-testing its retrieved evidence? Ordinary retrieval-plus-generation can appear convincing even when one paper supplies all support, small wording changes alter the sources, or contradictory evidence remains outside the initial Top-K.

Our system belongs to T1 because ranked retrieval is an inspectable IR component and generated factual statements are audited against retrieved corpus passages. The useful output is an answer plus an evidence audit, rather than a citation-decorated answer alone. The intended user is a student or researcher exploring a bounded scientific corpus.

### Project contribution and research context

The contribution is a novel combination and implementation for this project: claim-level checks, full-source removal with reretrieval, deliberate counter-evidence search, query wording probes, and transparent decision gates. Hybrid retrieval and citation checking are assignment sample ideas and are not claimed as novelty. SciFact [1], BEIR [2], RRF [3] and FActScore [4] provide context. We do not claim that robustness, contradiction retrieval or abstention are new research topics.

### Assignment alignment

| Criterion | Marks | Implementation evidence |
| --- | --- | --- |
| IR principles | 30 | Postings, TF/DF/IDF, BM25, TF-IDF cosine, heap Top-K, dense cosine, source-level RRF |
| Working system | 20 | Real corpus/queries, local generation, Streamlit inspector and reproducible CLI |
| Evaluation | 15 | Held-out qrels, baseline comparisons, raw runs, novelty diagnostics |
| Novelty / track | 15 | Counterfactual evidence testing of trustworthy RAG |
| Report / video | 20 | Required sections, graph/table, work-division template, live no-slide video plan |

The official assignment permits partial implementations and declared AI use. It requires a GitHub link, a 5-8 minute live explanatory video with no slides, and a report of at most eight main pages excluding references/appendix. It does not require deployment. The actual announcement time/deadline must be confirmed by the team.

## 2. How IR is used

```text
Question -> BM25 + dense -> RRF / Top-K -> LLM / claims
 -> source ablation + counter-search + query probes + NLI
 -> KEEP / QUALIFY / REWRITE / ABSTAIN
```

Figure 1. The generator cannot replace the retriever. Stress tests return to the same ranked IR pipeline and inspect actual retrieved evidence.

| IR concept | Choice and purpose | Code |
| --- | --- | --- |
| Document definition | One title + abstract per source. Dense passages are sentence windows; qrels and trust counts use unique sources. | preprocessing.py / data.py |
| Tokenization / normalization | NFKC, case folding, alphanumeric biomedical tokens. Keep negation; no unsupported stemming/phrase-search claims. | preprocessing.py |
| Dictionary / postings / DF / IDF | Own postings map stores source TF; document lengths and DF support interpretable sparse scores. | sparse_retriever.py |
| BM25 / heap Top-K | k1=1.2, b=.75; term discrimination and length normalization; heap selection among posting matches. | sparse_retriever.py |
| TF-IDF / cosine | Sublinear TF and smoothed IDF, L2 normalization. A second simple baseline reveals term-vector behavior. | sparse_retriever.py |
| Dense cosine | Normalized MiniLM embeddings; max cosine over source windows. Semantic retrieval supplements lexical matching. | dense_retriever.py |
| Hybrid rank fusion | RRF(c=60), candidate pools 50, output 5; sum 1/(60+rank). No raw-score scale assumptions. | hybrid_retriever.py |

BM25 uses log(1+(N-df+.5)/(df+.5)) multiplied by tf*(k1+1)/(tf+k1*(1-b+b*dl/avgdl)). TF-IDF cosine and dense cosine are relevance scores, not evidence entailment. The inspector exposes source/chunk IDs, rankings, available component scores, DF, IDF and posting samples.

Dense windows contain at most 150 words with one-sentence overlap. The encoder applies its tokenizer limit as well; dense source scores are max-pooled, so overlapping windows do not become extra independent source votes. Caches are fingerprinted by model configuration and corpus windows.

## 3. Beyond IR and evidence stress tests

Flan-T5-small generates a short evidence-conditioned answer locally. Optional remote chat providers are configurable through environment variables. Initial source references are provisional context IDs; final source references are assigned by the audit. On generator failure, an explicit extractive fallback replaces it, never a fake LLM answer.

Claim extraction splits sentence/semicolon candidates, bounded to three. This is a deliberately modest parser: a complex sentence may not be atomic. Extra sentences are omitted from final output rather than left unverified. A separate pretrained DeBERTa NLI model scores retrieved sentences/full passages as support, contradiction or neutral. Missing named-gene/entity anchors prevent a decisive label. It is not the generator judging itself; its scientific-domain errors remain a limitation.

| Component | Operational definition | Interpretation |
| --- | --- | --- |
| Support S | Maximum entailment score and actual quotes, using up to three matched sentences plus full passage. | Scores >=.80 are heuristic support, not probability of truth. |
| Source survival A | Exclude strongest supporting sources one at a time, including exact duplicate abstracts; reretrieve original + claim query. | Fraction with retained support; max three removals. Distinct IDs do not prove independence. |
| Query stability Q | Mean Jaccard of source Top-K against three wording probes. Optional reviewed rewrites accepted. | Empty results score zero; templates are not validated semantic paraphrases. |
| Counter-evidence C | Claim plus three contrary expansions through real retriever; classify only actual corpus quotes. | No contradictory passage can be invented from model memory. Bound audit to 15 passages. |

### Interpretable robustness decision

We retain the vector (S, A, Q, C, distinct source groups, query relevance). There is no arbitrary weighted aggregate or calibrated truth score. Missing NLI or weak relevance causes ABSTAIN. Support plus contradiction causes QUALIFY/CONTESTED. Contradiction alone causes REWRITE to an attributed source quote. No adequate support causes ABSTAIN. One support group, incomplete ablation survival, or Q<.40 causes QUALIFY. Otherwise KEEP means robust only under the tested corpus and perturbations.

The query relevance gate is top dense cosine >=.30 where dense is available; sparse-only uses known vocabulary as a weaker proxy. A decisive passage must retain >=.25 of the claim's IDF-weighted term mass. This topical/entity safeguard was added during live error analysis and may reject synonyms. Retrieval settings and the main stance/stability gates preceded held-out retrieval runs; the safeguards are exploratory. KEEP cannot establish that all literature was searched or that studies are independent.

Live questions of the form Does X reduce/increase/improve/prevent Y receive active/passive rewrites preserving subject, object and effect. Other inputs use wording wrappers. The published claim-query benchmark uses wrappers; neither mode is claimed to be human-validated paraphrasing.

## 4. Retrieval evaluation

BEIR SciFact supplies 5,183 abstracts and 300 test queries with relevance judgments. We evaluate source-level rankings using all test qrels. Macro P@5 divides by five even when fewer results are returned; Recall@5 uses each query's positive qrels; MRR@5 is explicitly truncated; nDCG@5 accounts for rank. Documents unjudged in qrels receive zero relevance in these benchmark metrics, not a scientific-falsity label.

| Method | P@5 | Recall@5 | MRR@5 | nDCG@5 |
| --- | --- | --- | --- | --- |
| tfidf | 0.1513 | 0.7011 | 0.5607 | 0.5888 |
| bm25 | 0.1487 | 0.6911 | 0.5971 | 0.6161 |
| dense | 0.1600 | 0.7320 | 0.5967 | 0.6251 |
| hybrid | 0.1647 | 0.7526 | 0.6196 | 0.6487 |

Actual evaluated sources: 5183; queries per method: 300; dense windows: 10060. Seed: 358. Raw runs and per-query CSVs are included in results/.

| Hybrid minus BM25 | Mean delta | Paired bootstrap 95% interval |
| --- | --- | --- |
| P@5 | +0.0160 | [+0.0080, +0.0240] |
| Recall@5 | +0.0614 | [+0.0272, +0.0969] |
| MRR@5 | +0.0226 | [-0.0062, +0.0501] |
| nDCG@5 | +0.0326 | [+0.0063, +0.0595] |

These are observed results, not guaranteed improvements. The dense model is not fine-tuned here. The bootstrap resamples paired queries 2,000 times; it estimates variation across these queries rather than universal generalization. MRR delta uncertainty includes zero. Recorded times reuse query caches, so they are not an unbiased latency comparison.

The experiment scripts contain no test-set hyperparameter search. Public gold is loaded by evaluation commands only. Runtime trust decisions receive passages and model predictions, never gold relevance or stance labels. Independent team judgments remain pending in docs/HUMAN_REVIEW.md; they must not be fabricated.

## 5. Novelty evaluation and limitations

Fifty test queries were sampled with seed 358; the exact IDs are recorded. Wording stability, contradiction retrieval and gold-source removal were measured across BM25, dense and hybrid.

| Method | Jaccard | Contradiction recall: K | Expanded 4 searches | Original at 4K |
| --- | --- | --- | --- | --- |
| bm25 | 0.674 | 0.788 | 0.788 | 0.879 |
| dense | 0.760 | 0.818 | 0.909 | 1.000 |
| hybrid | 0.719 | 0.788 | 0.788 | 0.909 |

| Method | Source-removal queries | With gold alternative | Alternative recall |
| --- | --- | --- | --- |
| bm25 | 35 | 5 | 0.300 |
| dense | 33 | 4 | 1.000 |
| hybrid | 37 | 5 | 0.500 |

Counter recall uses only original-dev annotated contradictory sources among sampled queries. Four searches consume more retrieval slots than K; original-at-4K is a budget-matched retrieval-slot control, but union unique-source counts can differ. These are retrieval recall diagnostics, not end-to-end contradiction-classification accuracy. Gold ablation excludes the best retrieved relevant source; absence of an annotated alternative is not proof of falsity.

Isolated NLI diagnosis: 50 original-dev gold rationale pairs; thresholded label accuracy 0.260; decisive coverage 0.340; accuracy on decisive pairs 0.765. Abstentions count as incorrect in the overall accuracy denominator. Gold rationale text is supplied only for this diagnosis, never runtime. This is not end-to-end answer accuracy. The model was changed after an initial MiniLM2 domain failure; this stance validation is exploratory.

### Limitations and next steps

SciFact is a bounded corpus and often has one annotated evidence source. Generic NLI can fail on biomedical quantities, populations, negation and terminology. High thresholds may abstain frequently; abstention alone is not evidence of quality. Max-over-sentence aggregation can admit a false positive. Query probes are simple wrappers. Exact duplicate detection does not establish study independence. Claim candidates may be incomplete or compound. The small generator may produce fragments or overgeneralizations; raw output remains visible.

Human review should compare initial ordinary-RAG answers with final accepted claims, reporting unsupported-claim rate AND accepted-claim coverage. This review is pending. A future course-project milestone can add train-only threshold calibration, verified paraphrases, a domain-trained stance model with clean evaluation splits, and a larger evidence corpus. No claim is made that the current system establishes clinical truth.

## 6. Working demo, work division and AI use

The repository provides CLI experiments and a Streamlit demo. The main panel shows the final decision and answer; the inspector exposes real source windows and scores. Claim panels show quotes, NLI values, counter-search runs and leave-one-source-out reretrieval. A complete JSON audit is downloadable. Synthetic unit-test fixtures are never substituted for corpus retrieval or evaluation.

Example live-run artifact: query "What potential does antiretroviral therapy have to prevent HIV-associated tuberculosis?"; generation local-flan-t5-small; final decision QUALIFY. The exported run records initial answer, evidence, queries, source exclusions, scores and runtime. This is a recorded result of the pipeline, not a hard-coded app response.

### Reproducibility and submission

Install requirements in a Python 3.12 environment, download the two public data archives, run sparse evaluation, full evaluation and model warmup, then launch Streamlit. Dataset checksums, model identities, dependency lock snapshot and raw experiment files are saved. API keys, if used, are environment variables and are not committed. First CPU indexing takes several minutes; later runs reuse cached embeddings.

The 7-minute no-slide demo plan covers T1 problem, live query, postings/weights, stress tests, measured baseline comparison, one limitation and each member's actual owned component. Recording/uploading and GitHub publication remain team tasks unless separately supplied. This draft does not invent a repository URL or video link.

### Work division - complete with actual contributions

| Member | Actual work | Video ownership |
| --- | --- | --- |
| [Member 1] | [Team to confirm] | [Team to confirm] |
| [Member 2] | [Team to confirm] | [Team to confirm] |
| [Member 3] | [Team to confirm] | [Team to confirm] |
| [Member 4, if any] | [Team to confirm] | [Team to confirm] |

### AI-use declaration

OpenAI Codex assisted with requirements interpretation, design, original code, tests/debugging, experiment scripts, documentation and this report. The human team must review and explain the submission. Runtime pretrained models are MiniLM embeddings, DeBERTa-v3-small NLI and Flan-T5-small generation; earlier diagnostics used MiniLM2 NLI. An optional remote provider must be declared if actually used. Public libraries/datasets/models are credited; no existing project implementation was cloned or submitted as original work.

Final checklist: confirmed 36-hour deadline; real team contributions; independently judged queries; GitHub README/repository link; 5-8 minute live video link; updated report PDF with these placeholders resolved. No percentages are needed for work division.

## References (excluded from main-page limit)

[1] Wadden et al. (2020). Fact or Fiction: Verifying Scientific Claims. https://arxiv.org/abs/2004.14974 . Data/schema: https://github.com/allenai/scifact/blob/master/doc/data.md . Annotation license CC BY 4.0; corpus ODC-By 1.0.

[2] Thakur et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. https://arxiv.org/abs/2104.08663 . Dataset distribution: https://github.com/beir-cellar/beir .

[3] Cormack, Clarke and Buettcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods. https://doi.org/10.1145/1571941.1572114 .

[4] Min et al. (2023). FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. https://arxiv.org/abs/2305.14251 .

[5] Sentence Transformers all-MiniLM-L6-v2 model card. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 . Apache-2.0.

[6] Sentence Transformers cross-encoder/nli-deberta-v3-small model card. https://huggingface.co/cross-encoder/nli-deberta-v3-small . Earlier diagnosis: https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768 . Both Apache-2.0; output order contradiction, entailment, neutral.

[7] Google Flan-T5-small model card. https://huggingface.co/google/flan-t5-small . Apache-2.0.

[8] Wadden et al. (2022). SciFact-Open: Towards open-domain scientific claim verification. https://arxiv.org/abs/2210.13777 .

[9] CSD358 IR Hackathon Midsem Assignment 2026. Official eight-page PDF supplied by the user; primary source for track, rubric and submission requirements.

### Library credit

NumPy/SciPy: matrices; scikit-learn: TF-IDF; Sentence Transformers/PyTorch/Transformers: pretrained inference; Streamlit: interface; Matplotlib: evaluation figures; pytest: functional tests; truststore: validated system TLS trust. Python standard library implements our postings/BM25/heap/RRF and data serialization. ReportLab/PDFium support report creation/verification. Exact installed versions are in requirements-lock.txt.
