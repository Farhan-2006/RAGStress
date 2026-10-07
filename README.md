# RAGStress / TrustRAG

A lightweight evidence-auditing layer for Retrieval-Augmented Generation (RAG). Given a question (or a claim), it retrieves scientific abstracts, drafts an answer with a small local model, and then **audits each claim** against supporting evidence, counter-evidence and query-wording stability. Each answer ends with one decision: `KEEP`, `QUALIFY`, `REWRITE` or `ABSTAIN`.

It runs on a CPU, needs **no API key**, and uses the public BEIR / SciFact dataset.

---

## 1. Requirements

| Item | Needed |
|---|---|
| Python | 3.12 recommended (3.10+ should work) |
| Disk space | about 3 GB (PyTorch, models, dataset, cached embeddings) |
| RAM | 8 GB or more recommended |
| Internet | Needed once, for pip packages, the dataset and the models. Afterwards it runs offline. |
| GPU | Not required |

---

## 2. Setup (one time)

Open a terminal in the project folder, the one containing `app.py`, then run the commands for your OS.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m src.cli download
.\.venv\Scripts\python -m src.cli warmup
```

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.cli download
.venv/bin/python -m src.cli warmup
```

What each step does:

| Step | Purpose |
|---|---|
| `pip install -r requirements.txt` | Installs PyTorch, transformers, sentence-transformers, Streamlit and the other libraries. Use `requirements-lock.txt` instead to reproduce the exact tested versions. |
| `python -m src.cli download` | Downloads the SciFact data into `data/`. |
| `python -m src.cli warmup` | Builds the search indexes and downloads and caches the models into `cache/`. The first run takes several minutes. |

Models downloaded automatically from Hugging Face:
`sentence-transformers/all-MiniLM-L6-v2` (dense retrieval), `cross-encoder/nli-deberta-v3-small` (claim verification) and `google/flan-t5-small` (answer drafting).

> **Tip:** On macOS/Linux, you can instead run `source .venv/bin/activate`. Then you can type `python` instead of `.venv/bin/python` in every command below.

---

## 3. Run the web app (main demo)

### Windows
```powershell
.\.venv\Scripts\python -m streamlit run app.py
```
or run the helper script:
```powershell
.\Launch-Demo.ps1
```
(If PowerShell blocks scripts, run `Set-ExecutionPolicy -Scope Process Bypass` first.)

### macOS / Linux
```bash
.venv/bin/python -m streamlit run app.py
```

Then open **http://localhost:8501** if the browser doesn't open automatically.

### Using the app

1. Choose **Ask a question** or **Check a claim**.
2. Type your input. Example question: *What potential does antiretroviral therapy have to prevent HIV-associated tuberculosis?* Example claim: *Aspirin reduces the risk of colorectal cancer.*
3. Click **Check answer** / **Check claim**. The first request loads the models, so it is slower. Later ones take roughly 10 to 15 seconds on a CPU.
4. Read the result:
   - Final answer and decision (KEEP / QUALIFY / REWRITE / ABSTAIN) with the reason
   - **Claims & evidence**: supporting, conflicting and inconclusive passages
   - **Stress tests**: counter-evidence search and wording-variation checks
   - **Approved source documents**: the documents actually used
   - **Retrieval details**: ranks, scores and passages
5. Use **Advanced settings** to change the retrieval method (Hybrid, BM25, TF-IDF or Dense), Top-K, or the generator.
6. Export the run as Markdown or JSON from the results page.

Generator options: **Local model** (default, no key), **Retrieved quotations** (extractive), and **External API** (optional, see section 7).

---

## 4. Run from the command line (no UI)

Run these from the project root.

```bash
# Ask a question; writes the full audit to results/latest-audit.json
python -m src.cli ask --query "Does vitamin D reduce the risk of fractures?"

# Check a claim instead of a question
python -m src.cli ask --claim-mode --query "Statins increase the risk of diabetes."

# Options: --method bm25|tfidf|dense|hybrid   --k 5   --generator local|extractive|remote
#          --sparse-only (skip dense model)   --output path/to/file.json
```

Other commands:

```bash
python -m src.cli evaluate --split test        # IR evaluation: TF-IDF, BM25, Dense, Hybrid (P@K, Recall, MRR, nDCG)
python -m src.cli novelty --limit 50           # Wording-stability and counter-evidence retrieval experiment
python -m src.cli stance --split test          # NLI classifier diagnostic on annotated rationale pairs
```

Outputs are written to `results/`.

---

## 5. Run the tests

```bash
python -m pytest -q
```

Optional UI smoke test. It renders the app against recorded sanity outputs and needs no models:

```bash
python scripts/verify_ui.py
```

---

## 6. Reproduce the 50-case evaluation

```bash
python run_evaluation.py                # resumes if a compatible checkpoint exists
python run_evaluation.py --no-resume    # full fresh rerun of vanilla, verification-only and full audit
python run_evaluation.py --limit 10     # quick run on fewer examples
python run_evaluation.py --report-only  # rebuild the report from completed results
```

- It uses 50 seeded SciFact dev examples. A full run takes a long while on a CPU, because it runs 3 systems per example.
- Output goes to `evaluation/results/`. Start with `final_report.md`, `comparison.csv` and the `plots/` folder.
- **Note:** The stored results in `evaluation/results/` describe the *earlier short-answer revision*. They are preserved in `evaluation/RAGStress_Evaluated_Snapshot.zip`. Use `--no-resume` to benchmark the current paragraph revision. `--report-only` refuses to relabel the older results.

Headline stored results (earlier revision): NLI groundedness 65.67% → 87.10%, mean latency 1.08s → 11.83s. Decisions: 26 KEEP / 6 QUALIFY / 16 REWRITE / 2 ABSTAIN. These are automated diagnostics, not verified correctness.

---

## 7. Optional: external LLM instead of the local model

Set these environment variables and select **External API** in the app (or `--generator remote`). The endpoint must be an OpenAI-style chat API. Never commit these values.

```powershell
# Windows PowerShell
$env:RAG_LLM_URL="https://your-endpoint/v1/chat/completions"
$env:RAG_LLM_MODEL="your-model-name"
$env:RAG_LLM_KEY="your-key"
```
```bash
# macOS / Linux
export RAG_LLM_URL="https://your-endpoint/v1/chat/completions"
export RAG_LLM_MODEL="your-model-name"
export RAG_LLM_KEY="your-key"
```

Optional: `RAG_NLI_MODEL` overrides the verification model (default `cross-encoder/nli-deberta-v3-small`).

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: Run python -m src.cli download first.` | Run the `download` step in section 2. |
| `ModuleNotFoundError` | The venv isn't active, or you ran `pip` outside it. Use `.venv/.../python -m pip install -r requirements.txt`. |
| Commands fail with `No module named src` | Run them from the project root, the folder containing `app.py` and `src/`. |
| First question is very slow | Normal: models are loading and embeddings are being built. Run `warmup` beforehand. |
| Banner says the evidence-checking model could not load | Check your internet connection, then rerun `python -m src.cli warmup`. Claims stay unverified (ABSTAIN) until the model loads. |
| SSL or certificate errors behind a corporate or college network | The project uses your OS trust store through `truststore`. Make sure `pip install -r requirements.txt` completed. Never disable TLS verification. |
| Port 8501 already in use | Add `--server.port 8502` to the streamlit command. |
| Out of memory | Use `--sparse-only` on the CLI, or the Retrieved quotations generator in the app. |
| `pip` can't find a compatible `torch` | Use Python 3.10 to 3.12, then reinstall. |

To reset everything, delete the `data/` and `cache/` folders and redo the setup. Both are git-ignored and regenerated automatically.

---

## 9. Project layout

```
app.py                 Streamlit web app (entry point)
run_evaluation.py      Paired 50-case evaluation runner
src/
  cli.py               Command-line entry: download | warmup | ask | evaluate | novelty | stance
  data.py              Dataset download and loading (SciFact)
  sparse_retriever.py  BM25 and TF-IDF
  dense_retriever.py   MiniLM dense retrieval
  hybrid_retriever.py  Reciprocal Rank Fusion (BM25 + dense)
  generator.py         Local FLAN-T5 / extractive / remote generation
  stance.py            NLI verification (DeBERTa)
  counter_evidence.py  Counter-evidence queries
  stability.py         Query-perturbation stability
  trust.py, claims.py, answer_repair.py, pipeline.py   Decision logic (KEEP/QUALIFY/REWRITE/ABSTAIN)
evaluation/            Study design, analysis, dataset and stored results
calibration/           Frozen thresholds and calibration data
tests/                 pytest suite
docs/                  Report, design notes, references
data/, cache/          Created automatically (dataset, models, embeddings)
results/               CLI outputs
```

## 10. Pipeline at a glance

Query → normalization → BM25 + dense retrieval → RRF fusion → Top-K passages → local FLAN-T5 draft → claim extraction → DeBERTa NLI verification → counter-evidence search + wording stability → **KEEP / QUALIFY / REWRITE / ABSTAIN**.

Neural labels can be wrong, and an exact quote is not independent proof of scientific truth. Treat the output as an evidence audit, not a guarantee.

## 11. Data, models and credits

- Data: BEIR SciFact and the original SciFact release. Checksums are in `dataset-manifest.json`. Licenses and papers are in `docs/REFERENCES.md`.
- Models: `all-MiniLM-L6-v2`, `nli-deberta-v3-small`, `flan-t5-small`.
- Authors: Farhan Naik, Hunar Bhatia, Aniket Sharma.
- More detail: `docs/DESIGN.md`, `docs/CALIBRATION.md`, `docs/EXPERIMENTS.md`, `docs/UI.md`.
