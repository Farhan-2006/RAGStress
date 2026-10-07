# RAGStress / TrustRAG

**Counterfactual evidence stress testing for retrieval-augmented generation.** CSD358 T1 hackathon prototype.
The core question: does a generated claim remain defensible after changing its evidence?

## Setup (Python 3.12 recommended, CPU sufficient)
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m src.cli download
.\.venv\Scripts\python -m src.cli evaluate --sparse-only
.\.venv\Scripts\python -m src.cli evaluate
.\.venv\Scripts\python -m src.cli warmup
.\.venv\Scripts\python -m streamlit run app.py
```
On macOS/Linux substitute `.venv/bin/python`. First model use downloads public pretrained weights; allow several minutes and about 2 GB total disk including dependencies. No GPU or API key required. Run from this repository root.

For the exact tested Python 3.12 dependency snapshot, install `requirements-lock.txt` instead of the version ranges. `requirements-report.txt` supplies optional report rendering tools. The local `Launch-Demo.ps1` also recognizes the environment prepared alongside this workspace; a fresh checkout should create its own `.venv`.

```powershell
python -m pytest -q
python -m src.cli ask --query "What potential does antiretroviral therapy have to prevent HIV-associated tuberculosis?"
python -m src.cli ask --claim-mode --query "Antiretroviral therapy increases rates of tuberculosis across a broad range of CD4 strata."
python -m src.cli novelty --limit 50
python -m src.cli stance --limit 50
```
Use your environment's Python in every command. `--sparse-only --method bm25 --generator extractive` runs without the dense/generator model; the NLI model is still needed for verification. If NLI fails, the system abstains and labels claims unverified. Dense failures are explicit; hybrid is never secretly BM25.

Optional remote chat completion provider: set `RAG_LLM_URL` to the complete compatible endpoint, `RAG_LLM_MODEL`, and `RAG_LLM_KEY` in environment, then `--generator remote`. Never commit credentials. This path must be separately verified with your provider; local mode is the default tested path.

## Architecture and IR choices
`question -> tokenization -> postings BM25 + normalized MiniLM windows -> source-level max-pooling -> RRF -> Top-K passages -> local Flan-T5 -> sentence claim candidates -> independent NLI -> source exclusion + counter-search + wording probes -> decision`

- Source/document: one scientific abstract with its title; corpus has meaningful distractors. Passage: bounded sentence windows (150 words, one-sentence overlap). Dense tokenizer still truncates at its model limit; title consumes tokens.
- NFKC/case folding; explicit biomedical/alphanumeric tokenization, negation retained. No unsupported phrase search or stemming claims.
- Own dictionary/postings, TF, DF, smoothed BM25 IDF and length normalization. Top-K uses a heap. Query term/postings inspection visible.
- TF-IDF baseline uses sublinear TF, sklearn smoothed IDF, L2 normalization; dot product equals cosine.
- MiniLM normalized embeddings and dot product cosine. Source score = maximum window cosine. Exact cached embeddings keyed by corpus/windows/model configuration.
- RRF: sum of `1/(60+rank)` over sparse/dense pools of 50. No addition of incompatible raw score scales. Missing pool scores/ranks remain absent (not zero); UI shows available scores.
- Gold relevance/stance annotations loaded only by experiment commands, never runtime verification.

## Stress tests and decisions
Claim candidates are sentence/semicolon splits, max three; complex sentences may remain non-atomic. Unexamined sentences are omitted from the final answer. Local generation aims for one short claim; small model limitations are exposed.

NLI checks up to three lexically relevant sentences and the full retrieved passage using a separate pretrained model with documented contradiction/entailment/neutral order. A named gene/entity missing from the passage prevents decisive labeling. Threshold **0.80** is a prototype operating point, not calibrated confidence. Model revisions are pinned; entity aliases can still cause conservative false negatives.

Counter-search retrieves the claim plus three contrary-evidence expansions; only actual corpus passages can be classified. All search strings/runs are downloadable. At most 15 unique passages are audited.

Leave-one-source-out tests the strongest supporting sources (max three), excludes all windows and exact duplicate abstracts, reretrieves original and claim queries, and measures surviving support. Different IDs do not establish scientific independence.

Query stability = mean Jaccard of unique Top-K source sets against three variants. Common "Does X reduce/increase/improve/prevent Y?" questions receive rule-based active/passive rewrites that preserve entities and the verb; other inputs receive template wording probes. Empty lists score zero. User-reviewed equivalent queries may be supplied; automatic variants are not claimed to be human-validated paraphrases. The published 50-query claim benchmark uses the template probes.

No weighted truth score. Component vector: support maximum, contradiction maximum, distinct-source groups, ablation survival, query Jaccard, query relevance. Decision gates:
1. Missing classifier, weak query relevance, or no support/no contradiction -> ABSTAIN.
2. Both support and contradiction -> QUALIFY (contested).
3. Contradiction only -> REWRITE as an attributed corpus quote.
4. Single support group, failed source survival, or Jaccard <0.40 -> QUALIFY.
5. Otherwise KEEP, explicitly robust only within tested corpus/perturbations.
Query relevance uses top dense cosine >=0.30 where dense is available; this is another declared heuristic. Sparse-only uses known vocabulary and is weaker. A passage must retain at least 0.25 of the claim's IDF-weighted term mass for a decisive stance; this conservative topical gate limits unrelated NLI false positives but may reject valid synonyms. It was added during live error analysis, not calibrated for a claimed statistical guarantee.

## Evaluation and honesty
`results/` holds actual per-query scores, raw ranked runs, summary tables, graphs and protocol notes. Binary BEIR relevance says nothing about whether evidence supports or contradicts a claim. Original SciFact dev gold provides separate contradiction retrieval and NLI diagnosis.

Measured on all 300 BEIR test queries / 5,183 source abstracts:

| Method | P@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.1513 | 0.7011 | 0.5607 | 0.5888 |
| BM25 | 0.1487 | 0.6911 | 0.5971 | 0.6161 |
| Dense | 0.1600 | 0.7320 | 0.5967 | 0.6251 |
| Hybrid | 0.1647 | 0.7526 | 0.6196 | 0.6487 |

Hybrid-minus-BM25 Recall@5 delta 0.0614 (paired-bootstrap 95% interval 0.0272–0.0969); nDCG@5 delta 0.0326 (0.0063–0.0595). MRR improvement interval includes zero. Runtime timings are warm-cache measurements with query embeddings reused by hybrid, so they are not a fair latency comparison.

On 50 seeded test queries, mean template-probe Jaccard is BM25 0.6741, dense 0.7599, hybrid 0.7190. For 11 queries with annotated counterevidence, hybrid counter-search does not improve recall over initial K (both 0.7879); original-at-4K obtains 0.9091. Our novelty is a testable audit, not a claimed counter-retrieval performance win. Original-dev isolated DeBERTa NLI diagnosis is 13/50 correct decisive labels, coverage 17/50, accuracy among decisive labels 13/17. These measurements do not prove answer truthfulness.

Retrieval metrics: macro P@5, Recall@5, truncated MRR@5, nDCG@5. Precision denominator stays K even with fewer results. Unique source IDs prevent duplicated windows inflating recall. Paired bootstrap estimates uncertainty of hybrid-minus-BM25 differences. No guarantee hybrid wins.

Novelty: seeded 50-query wording probes across retrievers; gold-aware source-removal diagnostic with explicit counts of available alternatives; counter-search recall against annotated contradictory sources. Counter expansion gets more retrieval slots, so recall gains are not equal-budget evidence of better ranking. Isolated NLI on gold rationale sentences is **not** end-to-end answer accuracy. Human judged queries and generated claims require the team to fill the worksheet before submission.

## Source files
`src/data.py`, `preprocessing.py`, `sparse_retriever.py`, `dense_retriever.py`, `hybrid_retriever.py`: IR.
`generator.py`, `claims.py`, `stance.py`: beyond-IR generation/audit.
`counter_evidence.py`, `ablation.py`, `stability.py`, `trust.py`, `pipeline.py`: stress-testing novelty.
`evaluation.py`, `cli.py`: reproducible experiments. `app.py`: Streamlit.

## Data, models and credits
BEIR SciFact archive: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
Original SciFact data only: https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
Checksum manifest: `data/manifest.json`. Original annotations CC BY 4.0; abstracts ODC-By 1.0 (SciFact license). Credit dataset authors; redistribution should preserve licenses.
Models: `sentence-transformers/all-MiniLM-L6-v2`, `cross-encoder/nli-deberta-v3-small`, `google/flan-t5-small`. Earlier diagnostics also used `cross-encoder/nli-MiniLM2-L6-H768`; set `RAG_NLI_MODEL` to that name to reproduce it. Apache-2.0 model cards; see `docs/REFERENCES.md`.
Libraries declared in requirements and lock snapshot. Code authored for this project with OpenAI Codex assistance; no existing project cloned/copied.

## Submission status
See `docs/REQUIREMENTS.md`, `DESIGN.md`, `EXPERIMENTS.md`, `AI_USE.md`, `VIDEO_PLAN.md` and `HUMAN_REVIEW.md`.
The final submission needs team names/actual work division, completed human judgments, a GitHub repository URL and a recorded 5-8 minute video link. Those cannot be fabricated. Keep the report <=8 main pages excluding references/appendix. No slides in video. Local execution is sufficient.

## Verified demo inputs
- Question (local LLM + hybrid): `What potential does antiretroviral therapy have to prevent HIV-associated tuberculosis?` -> QUALIFY; actual answer supported by source 4883040 but fragile under removal.
- Atomic claim mode: `Antiretroviral therapy has substantial potential to prevent HIV-associated tuberculosis.` -> QUALIFY; inspect source ablation.
- Atomic claim mode: `Antiretroviral therapy has no potential to prevent HIV-associated tuberculosis.` -> REWRITE to the attributed supporting corpus quote; inspect actual counterevidence.
- Question: `What is the certified wingspan of the invisible blue dragon on planet Zorblax-99?` -> ABSTAIN.
- Question: `Does antiretroviral therapy reduce tuberculosis incidence?` -> REWRITE. When generated wording fails verification, the system finds a relevant exact quotation in the already ranked sources, audits that replacement, and explicitly attributes it. The replacement remains marked fragile if it depends on one source.
- Limitation: NLI makes false contradiction predictions on some unrelated neural-development passages; show errors rather than claiming a perfect verifier.
Saved audits document past real runs; the app computes every new query, never returns canned demo outputs.

## Answer correction policy
Failed generation no longer automatically discards usable retrieved evidence. In question mode, replacement candidates must pass query-term coverage and semantic relevance gates and the same support/counterevidence/ablation audit. Unknown query terms retain high weight, preventing unrelated quotations from satisfying nonsense queries. The final answer identifies the correction and preserves the rejected generated claims in the inspector. These recovery gates are prototype heuristics; support for an exact quote does not establish its truth. Direct claim mode keeps evaluating the supplied claim and never substitutes a different claim. No support thresholds were lowered for this fix.
