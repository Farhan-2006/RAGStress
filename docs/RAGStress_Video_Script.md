# RAGStress live-demo video script
Target length: 7 minutes. No slides.

Preparation: confirm actual ownership before using this role allocation. Each speaker describes the component they genuinely own. Warm the local models and index before recording; keep all answers computed live. Open the app, relevant source files, stored CSVs and plots. Hide secrets. Keep all three queries ready to paste. Rehearse at roughly 125-140 words per minute, allowing processing pauses. The 7-minute timeline is a target; trim narration if real runs are slow and remain within 5-8 minutes. Never use slides, fake outputs or a promise that a query must pass.

- Question: What potential does antiretroviral therapy have to prevent HIV-associated tuberculosis?
- Opposing claim: Antiretroviral therapy has no potential to prevent HIV-associated tuberculosis.
- Limitation question: What is the certified wingspan of the invisible blue dragon on planet Zorblax-99?

## 0:00-0:40 | Farhan | Problem and track

**On screen:** Show the running Streamlit app. Keep this problem introduction under one minute.

**Narration:** We are Farhan, Hunar and Aniket, and our project is RAGStress for Track T1: Retrieval-Augmented Generation and Trustworthy Answers. A RAG answer may sound convincing even when it misreads a paper or overlooks disagreement. Our question is: does the answer remain defensible when we challenge its evidence? We combine real sparse and dense retrieval, then audit claims, search for counter-evidence and test sensitivity to query wording. The outcome can be keep, qualify, rewrite or abstain. I will demonstrate the retrieval component.

## 0:40-1:40 | Farhan | Real end-to-end question

**On screen:** Select Ask a question. Paste the ART question below and run it live. Show the computed paragraph, decision, Approved source documents and one source excerpt. Allow real processing time; never replace the result with a recording of a fabricated run.

**Narration:** Here is a real question about antiretroviral therapy and HIV-associated tuberculosis. The system searches our corpus of 5,183 SciFact abstracts. These are the retrieved documents, and here is the final paragraph. The approved documents are separate from the answer so the paragraph stays readable. They are documents actually used for retained content or a correction, not every document that retrieval happened to return. We can open the evidence and compare it with the claim. The initial answer is generated locally from retrieved passages by FLAN-T5-small; it is not a prewritten response and does not require an API key.

## 1:40-2:20 | Farhan | Pipeline and actual IR

**On screen:** Open Retrieval details. Show BM25/dense/RRF scores and the index dictionary with an actual query term. Briefly open src/preprocessing.py, sparse_retriever.py and hybrid_retriever.py in the editor. Point to postings and RRF rather than reading whole files.

**Narration:** Our pipeline normalizes the query, retrieves with BM25 and dense embeddings, fuses ranks, generates a draft, and audits the evidence. This is our real inverted index: term frequencies, document frequencies and IDF feed BM25. Dense retrieval uses cosine similarity over sentence windows and ranks unique sources. RRF combines ranks because BM25 and cosine scores have different scales. The source IDs and scores remain inspectable. We also retain TF-IDF and BM25 as simpler baselines. Hunar will now show what happens after retrieval.

## 2:20-3:20 | Hunar | Claim-level support and decisions

**On screen:** Show Claims & evidence for the actual question. Open src/claims.py and src/stance.py briefly. Point to one claim, support score, matched excerpt and alignment result.

**Narration:** I will demonstrate evidence auditing. The draft is split into sentence and semicolon claim candidates. A separate DeBERTa inference model checks retrieved evidence for support, contradiction or neutrality. This is not the generator simply declaring itself correct. Topic, entity and displayed-context checks also matter. Strong support uses a prototype threshold of point eight; contradiction requires point nine and context confirmation. Moderate support can qualify the answer. We keep the component signals instead of presenting an arbitrary truth percentage. These checks are fallible, and one strongly relevant source can be sufficient; a KEEP label does not prove scientific consensus.

## 3:20-4:10 | Hunar | Counter-evidence and correction

**On screen:** Select Check a claim and paste the opposing ART claim below. Run live. Read the actual decision. Show counter-search queries, ranked sources and any confirmed opposing excerpt. If the decision differs, describe it honestly; do not promise REWRITE.

**Narration:** Now I will test the opposite claim. Counter-search creates additional queries intended to find conflicting findings, but every result comes from our actual corpus retrieval. Negative wording in a query is not itself contradictory evidence. The verifier assesses the retrieved passage and its context. If the evidence contradicts the supplied statement, the system can rewrite it using an audited finding. If evidence is mixed it can qualify, and if aligned evidence is absent it can abstain. The inspector shows exactly which passages led to the decision.

## 4:10-4:40 | Hunar | Query stability and scope

**On screen:** Open Stress tests. Show actual variants, their Top-K IDs and Jaccard overlaps. Mention that source ablation is absent.

**Narration:** The second stress test changes query wording and retrieves again. Jaccard overlap measures how many Top-K source IDs are shared. Very low mean overlap can qualify an answer, but moderate variation is not automatically a failure. These controlled variants are not guaranteed perfect paraphrases, so stability is a sensitivity signal rather than proof of correctness. The final implementation has these two stress tests; source ablation is not included.

## 4:40-5:20 | Aniket | Live limitation and abstention

**On screen:** Select Ask a question. Paste the fantasy query below and run live. Show the computed decision and why it was made. Never imply a guaranteed label on every future input.

**Narration:** I will demonstrate evaluation and limitations. This deliberately out-of-corpus question asks about an invisible dragon on a fictional planet. Our scientific abstract corpus cannot establish that fact. The small generator may still produce a fluent draft, which is why the evidence audit matters. We show the actual output and its reasons rather than treating fluency as correctness. Other limitations include abstracts rather than full papers, imperfect claim splitting, generic-domain inference and incomplete relevance judgments. Abstention protects against some unsupported answers, but we have not established a general abstention accuracy score.

## 5:20-6:30 | Aniket | Real evaluation and baseline

**On screen:** Open Evaluation or the stored retrieval-test.csv and comparison.csv, then the real plots. Show historical-version notice. Briefly show evaluation/study.py or src/evaluation.py. These are working artifacts, not slides.

**Narration:** Our ranked-retrieval experiment uses 300 BEIR queries with public relevance judgments, a justified alternative to inventing manual labels. At K equals five, BM25 precision is point one four eight seven and recall point six nine one one. Hybrid reaches point one six four seven and point seven five two six. nDCG improves from point six one six one to point six four eight seven. A separate fifty-case historical experiment compares the same initial retrieval and generation with and without auditing. Model-assessed groundedness rises from sixty-five point six seven to eighty-seven point one zero percent; answer coverage falls to ninety-six percent. Mean latency rises from one point zero eight to eleven point eight three seconds. Verification-only matches full auditing at eight point nine eight seconds, so we have not established an additional quality benefit from stress tests. These are model diagnostics, not verified free-form answer accuracy. The fifty-case numbers belong to the earlier short-answer snapshot; the current paragraph revision needs a fresh benchmark.

## 6:30-7:00 | Aniket | Reproduction, integrity and next steps

**On screen:** Show README setup, requirements and references/AI-use notes. End on the working app or repository. All three members have now spoken; no slides are used.

**Narration:** The README records setup, models, data and evaluation commands. Local inference requires no API key. We used public datasets and pretrained models, and declare Codex assistance with code, debugging and documentation. The next steps are rerunning the current version, obtaining independent human judgments and testing more mixed or unanswerable cases. Our contribution is an inspectable combination of retrieval and evidence challenges, with honest limits and reproducible results. Thank you.

## Recording checklist
Problem and T1 relevance under one minute; at least one real end-to-end input and one limitation; whole pipeline followed by evaluation; actual postings, weights, scores or code visible; baseline P@5/Recall@5 and real evaluation graphs/tables; all three members explain confirmed ownership; historical results labelled; AI use declared; final runtime 5-8 minutes; unlisted link accessible to graders.