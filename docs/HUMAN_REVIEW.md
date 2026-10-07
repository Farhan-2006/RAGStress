# Required human judgment worksheet

Public qrels are objective benchmark judgments; they do not replace the team's independent error analysis.
Use at least 5 real queries, including support, contradiction, wording sensitivity and an out-of-corpus query.

1. Run each query in BM25 and hybrid; export audit JSON and save Top-5 ranked source IDs.
2. Two team members independently judge source relevance (0/1), each generated claim's support/contradiction/insufficient status, quote fidelity and whether the decision is appropriate.
3. Mark unjudged passages explicitly. Resolve disagreement and record a short explanation, not just numbers.
4. Calculate P@5 = relevant Top-5 / 5. Recall requires an explicitly bounded relevant set or known qrels; do not claim exhaustive recall from five inspected results.
5. Record ordinary initial answer versus stress-tested final answer. Count unsupported accepted claims / accepted claims and accepted claims / total claims (coverage). All-abstain can reduce unsupported rate trivially; report coverage alongside it.

| Query | Method | Ranked source IDs | Relevance judgments | Initial unsupported claims | Final unsupported accepted claims | Accepted/total | Appropriate decision? | Reviewers / notes |
|---|---|---|---|---|---|---|---|---|
| [Real query] | BM25 | [From run] | [Human] | [Human] | [Human] | [Human] | [Human] | [Names] |
| [Same query] | hybrid | [From run] | [Human] | [Human] | [Human] | [Human] | [Human] | [Names] |

NOT COMPLETED by an AI pretending to be team reviewers. Automated NLI scores are not human ground truth. The submission report must state whether this review was performed.
