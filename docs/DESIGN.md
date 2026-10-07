# Frozen MVP and feasibility audit

## Must have
1. Download BEIR SciFact + original SciFact annotations; checksum provenance.
2. Abstract-level retrieval evaluation with source IDs. Dense sentence windows max-pooled to source; no chunk duplication of qrel hits.
3. Own postings BM25 (k1=1.2, b=.75), sklearn sublinear TF-IDF/cosine, MiniLM normalized dense cosine, RRF(c=60), pool 50 / output 5.
4. Early held-out retrieval evaluation. Exact raw runs, per-query tables and paired-bootstrap uncertainty.
5. Grounded answer with local Flan-T5-small or configured remote LLM. Clearly named extractive fallback on failure.
6. Bounded atomic-claim audit (max 3 per answer), distinct model NLI for support/contradiction; abstain if NLI unavailable.
7. Full-corpus source exclusion BEFORE Top-K; reretrieve original and claim query for each supporting source (max 3).
8. Corpus-only counter-search; stance from retrieved sentences, never LLM memory.
9. Query variants and Jaccard Top-5; distinguish template wording perturbations from validated semantic paraphrases.
10. Conservative KEEP/QUALIFY/REWRITE/ABSTAIN; raw scores and thresholds visible. Rewrite uses verified text with citations rather than a second uncontrolled LLM answer.
11. Streamlit inspector, tests, README, report draft, no-slide video script, credits and AI log.

## Nice to have
- Reviewed LLM paraphrases, dependency diagram, larger claim sample, threshold calibration on original train only.
- Rationale sentence recall, selective-risk curves, manually adjudicated generated-answer claims.

## Cut first
- Deployment, animations, framework agents, fine-tuning, broad crawling, universal scientific truth scoring.

## Risks fixed before implementation
- SciFact mostly has one relevant evidence source; ablation will often have no annotated alternative. Report this as a corpus limitation, not a success/failure of scientific truth. Count available alternatives explicitly.
- BEIR qrels encode relevance, NOT stance; original dev supplies SUPPORT/CONTRADICT gold. No gold labels in runtime.
- Distinct IDs are a proxy only: duplicates and related studies can remain dependent. Detect exact normalized duplicate abstracts; no assertion of causal independence.
- Negation query expansion alone may not retrieve counterevidence; include original claim query and expose all actual search strings. NLI can miss numeric contradictions: report errors.
- Small local generator/NLI are domain-mismatched; score is a heuristic audit, not calibrated truth probability or medical advice.
- Query templates change wrapper words, not verified paraphrases; include a reviewable paraphrase protocol and avoid overclaiming stability.
- Leave-one-out means excluding EVERY window of a source, not just removing its current displayed passage.
- Claim extraction may be imperfect; expose claims and allow claim mode with user-provided atomic claim.
- No API required for retrieval; no silently fabricated dense/LLM output on dependency failures.

## Suggested 36-hour allocation
0-5h corpus, sparse and evaluation; 5-10h dense/hybrid; 10-17h generation + NLI;
17-25h stress tests/evaluation; 25-30h integration/human judgments;
30-34h report/video rehearsal; 34-36h reproducibility and submission.

Actual course deadline and team members must be supplied before submission; no invented names or ownership.
