# External sources and credit ledger

No existing implementation was cloned. Dataset archives and pretrained weights only were downloaded.

1. Official CSD358 IR Hackathon Midsem assignment (2026), supplied PDF. Primary requirements source.
2. Wadden et al. (2020), *Fact or Fiction: Verifying Scientific Claims*. https://arxiv.org/abs/2004.14974 ; original data schema https://github.com/allenai/scifact/blob/master/doc/data.md ; licenses https://github.com/allenai/scifact/blob/master/LICENSE.md . Scientific claims/rationales annotated SUPPORT or CONTRADICT; cited docs are not automatically evidence.
3. Thakur et al. (2021), *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. https://arxiv.org/abs/2104.08663 ; dataset distribution https://github.com/beir-cellar/beir . We load its SciFact corpus/queries/qrels directly; do not use copied BEIR project code.
4. Cormack, Clarke and Buettcher (2009), *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*. https://doi.org/10.1145/1571941.1572114 . RRF combines rankings without assuming raw score calibration.
5. Dense model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 . Normalized sentence embeddings; Apache-2.0. Model max-length truncation is disclosed.
6. NLI model card: https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768 . SNLI/MultiNLI-trained; documented output order contradiction, entailment, neutral; Apache-2.0. Biomedical transfer performance must be measured here.
7. Generator model card: https://huggingface.co/google/flan-t5-small . Instruction-tuned seq2seq model; Apache-2.0. Small local model is a constrained prototype, not expert verification.
8. Min et al. (2023), *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*. https://arxiv.org/abs/2305.14251 . Related work on claim-level evaluation; no claim that atomic evidence checks originate here.
9. Wadden et al. (2022), *SciFact-Open: Towards open-domain scientific claim verification*. https://arxiv.org/abs/2210.13777 . Evidence coverage matters; our smaller SciFact corpus is not the full literature.
10. Final NLI model card: https://huggingface.co/cross-encoder/nli-deberta-v3-small . Documented contradiction/entailment/neutral order; Apache-2.0. Selected after a training-split diagnostic following the earlier MiniLM2 domain failure. Validation remains exploratory and coverage is low.

Library roles: NumPy/SciPy matrices; scikit-learn TF-IDF; PyTorch/Transformers generation and NLI; Sentence Transformers embedding inference; Streamlit UI; Matplotlib plots; pytest important functionality. Standard library handles own postings, BM25, heaps, RRF, downloads, JSON/CSV. ReportLab/PDFium are document-authoring/verification tools, not retrieval methods.

Novelty language: "A novel combination and implementation for this project that extends an ordinary RAG baseline by actively testing whether generated claims remain defensible under changes to retrieved evidence." It is not a priority claim over the research literature. Hybrid retrieval and citation checking are sample ideas in the official assignment.
