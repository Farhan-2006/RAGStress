# RAGStress / TrustRAG

A lightweight evidence-auditing layer for RAG. It checks generated claims against counter-evidence and query perturbations, rather than attempting to exhaustively prove them.

Repository: https://github.com/Farhan-2006/RAGStress.

## Setup and demo
Python 3.12 recommended; CPU supported. From this directory:
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m src.cli download
.\.venv\Scripts\python -m src.cli warmup
.\.venv\Scripts\python -m streamlit run app.py
```
On macOS/Linux use `.venv/bin/python`. `requirements-lock.txt` records the tested environment. First indexing downloads public models and builds reusable embeddings; no API key is needed. Optional external generation uses RAG_LLM_URL, RAG_LLM_MODEL and RAG_LLM_KEY, never committed. Local mode is the tested default. Launch-Demo.ps1 also supports the environment prepared beside this workspace.

Choose Ask a question or Check a claim and enter your input; all outputs are computed live. Applied input/settings edits mark prior results; failed runs retain explicitly labelled prior results. Keyboard submission: Tab to Check answer / Check claim, then Enter. Advanced settings retain real retrieval methods, Top-K and optional wording variants. Source buttons select the reader in Retrieval details. Markdown/JSON exports describe the displayed run.

## Active architecture
Current answers use one complete paragraph. Generation requests three explanatory sentences; up to six factual candidates can be audited. When the small local model still returns a short draft, up to two relevant retrieved findings can supply context after the same support/counter-evidence checks accept them. No unverified facts are added to reach a word count. Supplied claims receive a paragraph explaining the assessment; all final outputs use one paragraph.

The answer paragraph contains no document IDs or inline source references. A separate Approved source documents section lists the documents actually used for retained claims or correction, with titles, IDs, approved excerpts and reader buttons. Retrieval alone does not approve a document; conflicting and inconclusive evidence remains visible in the inspector. Readable and technical exports preserve this separation, with raw references retained as audit metadata.

Query -> normalization -> inverted-index BM25 + normalized MiniLM dense windows -> RRF -> ranked Top-K passages -> local FLAN-T5 draft -> sentence/semicolon claim candidates -> separate DeBERTa NLI verification -> counter-evidence search + wording stability -> KEEP / QUALIFY / REWRITE / ABSTAIN.

Exactly two stress tests are active: counter-evidence search and query perturbation. TF-IDF cosine and BM25 remain inspectable retrieval baselines. Abstracts are sources; bounded overlapping sentence windows are passages; unique source rankings avoid duplicated-window retrieval votes. BM25 k1=1.2, b=.75; RRF constant 60, rank pools 50. Dense embeddings and model revisions are cached/pinned.

## Calibrated decisions
Strong support >=.80; aligned moderate support >=.60 remains qualified. Contradiction requires >=.90, context confirmation and adequate topic/entity alignment. Topical coverage .15 is a secondary alignment route alongside semantic cosine .40; specific mismatched numbered entities still block decisive labels. Severe wording instability is Jaccard <.15; moderate variation does not reject strong support. One relevant supporting document can be sufficient.

KEEP: strong aligned support, no confirmed conflict, no severe instability. QUALIFY: moderate support, genuine model-assessed conflict, or severe instability. REWRITE: contradicted wording or audited quote recovery. ABSTAIN: insufficient/unrelated evidence or unavailable verification. No arbitrary weighted truth probability or target decision distribution. Neural labels can be wrong; exact quotes are not independent proof of scientific truth.

## Reproducible comparison
The stored 50-case comparison describes the earlier short-answer revision, preserved in `evaluation/RAGStress_Evaluated_Snapshot.zip`. It has not been rerun for the paragraph revision. The interface labels those results as historical. Run `python run_evaluation.py --no-resume` to benchmark the current revision; ordinary resume rejects old checkpoints instead of silently attributing their scores to the changed system.

```powershell
python -m pytest -q
python run_evaluation.py
```
The evaluation uses a fixed seeded set of 50 original SciFact dev examples with available SUPPORT/CONTRADICT gold, because the original test release lacks stance annotations. Calibration uses separate original-training rationale pairs. Both systems use identical corpus, generator, hybrid ranking and K=5. Initial IR metrics are expected to be identical by design. The NLI-only component also runs. Gold never enters either runtime system.

Outputs: `evaluation/evaluation_dataset.json`, frozen manifest under `calibration/`, raw paired records, comparison CSV/JSON, error examples, statistical results, report and plots under `evaluation/results/`. The report distinguishes actual gold retrieval metrics from model-based groundedness diagnostics. Gold does not exhaustively label new generated claims; unavailable correctness/evidence-precision metrics are explicitly marked unavailable. There is no invented human judgment or universal trustworthiness claim.

Completed run: all 50 examples across vanilla, verification-only and full auditing (150 timed executions). Common NLI groundedness: 65.67% -> 87.10%; answer coverage: 100% -> 96%; mean latency: 1.083s -> 11.834s. Full decisions: 26 KEEP / 6 QUALIFY / 16 REWRITE / 2 ABSTAIN. Verification-only matches these aggregate quality metrics at 8.983s, so incremental stress-test quality benefit is not established. Automated answer-stance agreement improves 42% -> 52%, but McNemar p=.3323 does not establish significance. Gold-SUPPORT diagnostic agreement worsens; see the full confusion matrices. These are automated evidence diagnostics, not verified answer correctness. Full results: `evaluation/results/final_report.md` and `comparison.csv`.

To rebuild the previous report, use `python run_evaluation.py --report-only` from the preserved evaluated snapshot. In the current revision this command refuses to relabel older results. To actually rerun all systems and replace current timings: `python run_evaluation.py --no-resume`.

## Sources and integrity
Data: BEIR SciFact https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip and original SciFact https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz . Dataset checksums: dataset-manifest.json and data/manifest.json. Credit/licensing and papers: docs/REFERENCES.md. Original annotations CC BY 4.0; abstracts ODC-By 1.0.

Models: sentence-transformers/all-MiniLM-L6-v2, cross-encoder/nli-deberta-v3-small, google/flan-t5-small. Libraries are declared in requirements. Original project code was written with OpenAI Codex assistance; no existing project was cloned. Actual contributions, human judgments and the 5-8 minute live demo recording remain team responsibilities. Report limit: 8 main pages excluding references/appendix; video uses no slides. IR and evaluation remain central to T1.

## Report and video script

- [Report PDF](docs/RAGStress_Report.pdf) and [editable report](docs/RAGStress_Report.md): seven main pages plus references, measured results and version disclosures.
- [Video script PDF](docs/RAGStress_Video_Script.pdf) and [editable script](docs/RAGStress_Video_Script.md): a seven-minute live demonstration with three speakers.
- Team: Farhan Naik (2410110422), Hunar Bhatia (2410110427), Aniket Sharma (2410110516). The balanced ownership allocation is proposed and must be confirmed against actual contributions before submission.
