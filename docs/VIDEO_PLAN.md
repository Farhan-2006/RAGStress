# 7-minute live demo plan (no slides)

0:00-0:45 — Member 1 introduces T1 and question: "Should RAG trust its answer after the strongest evidence is removed?" Keep Streamlit/code onscreen, not a slide.

0:45-1:45 — Member 1 types a real question. Show ranked source IDs, BM25/dense/fusion ranks and corpus passages. Change one question to demonstrate live input.

Verified first question: "What potential does antiretroviral therapy have to prevent HIV-associated tuberculosis?" It produces a local LLM answer and QUALIFY after the supporting-source removal test. Use claim mode for "Antiretroviral therapy has no potential to prevent HIV-associated tuberculosis." to demonstrate REWRITE from actual counterevidence. Saved audits are records only; run the pipeline live in the recording.

1:45-2:35 — Member 2 opens `sparse_retriever.py` and the query postings/DF/IDF inspector. Explain document choice, normalization, IDF, length normalization, heap Top-K and RRF score formula.

2:35-4:05 — Member 3 shows initial LLM answer, extracted claim candidates and separate NLI stance scores; opens source-removal audit, excluded IDs and reretrieved supporting passages. Explain that distinct IDs do not prove independence.

4:05-5:00 — Member 3 opens actual counter-search strings and retrieved contradictory evidence; show query wording probes and Jaccard. Demonstrate QUALIFY/REWRITE/ABSTAIN with real recorded examples confirmed before filming.

5:00-6:05 — Member 4 if present (otherwise Member 2) shows actual evaluation CSV/plot, baseline versus final metrics, novelty tables. State whether hybrid improved or regressed; no invented wins. Show the command/output proving a reproducible run.

6:05-6:45 — Show a limitation live: NLI misclassification, claim fragmentation or an unanswerable question. Explain sparse/dense score is relevance, not entailment; classifier can miss quantities/negation. Do not hide bad generation.

6:45-7:00 — Confirm README run instructions, credits, AI use and actual member ownership. End on live system. Record each member explaining the component actually owned.

Before recording: warm model cache, verify examples, use screen recording with readable text/audio, avoid keys onscreen. Submit unlisted YouTube/Drive link. No video has been recorded or uploaded by this repository.
