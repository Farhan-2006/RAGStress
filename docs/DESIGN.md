# Frozen design
Keep the existing sparse/dense/RRF retrieval and local generation architecture. Exactly two stress tests remain: corpus counter-search and query perturbation. Evidence verification is separate from generation, uses explicit retrieved passages, and retains support/contradiction/neutral scores, topical alignment and entity diagnostics.

The calibrated policy permits a strong relevant single-document finding, distinguishes moderate support, and reserves wording qualification for severe overlap loss. All thresholds are prototype heuristics; calibration and sanity traces are in calibration/. Numeric identity mismatches remain blocking. Context confirmation reduces sentence-without-context contradiction flags. Alias matching handles a small explicit list and acronym-family forms; it is not a general biomedical ontology.

Train calibration precedes a saved hash manifest. Evaluation uses the same fixed original-dev subset and initial retrieval/generation for vanilla, NLI-only and full auditing. No evaluation gold is a runtime input. Model-based groundedness is a diagnostic, not human factual correctness. A small local generator and generic NLI remain material limitations.
