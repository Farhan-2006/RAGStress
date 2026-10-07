# Verified requirements and rubric mapping

Primary source: CSD358 IR Mid-term Assignment 2026.pdf, all 8 pages read, 6 October 2026.
This document records assignment requirements separately from our engineering decisions.

| Official criterion | Marks | Project evidence / acceptance check |
|---|---:|---|
| Relevant IR principles | 30 | Defined abstract/source and sentence windows; normalization/tokenization; dictionary/postings, DF/IDF; TF-IDF cosine and BM25; heap Top-K; semantic cosine; RRF; visible scores |
| Working system | 20 | Live arbitrary queries, actual corpus, run instructions, explicit model failures; no fixed demo answers |
| Evaluation | 15 | Held-out BEIR SciFact qrels, baseline comparison, P@5 / Recall@5 / MRR@5 / nDCG@5; manually judged queries; novelty experiments |
| Novelty | 10 | Integrated counter-search, query perturbation, claim audits and calibrated decisions; hybrid/citations alone are not novelty |
| Report | 10 | PDF <=8 main pages excluding references/appendix; required sections, pipeline diagram, tables/graphs |
| Video | 10 | 5-8 minutes, live end-to-end, limitation, pipeline/code/intermediate scores, evaluation, each member speaking; NO SLIDES |
| Track relevance | 5 | T1: inspectable IR and every generated factual claim traced to ranked evidence |

## Official restrictions / deliverables
- Team of 3-4; build in the announced 36-hour window. Track announcement/deadline not provided here; do not invent it.
- One team submission: GitHub repository link + reproducible README, video link (unlisted YouTube/Drive), report PDF.
- README must include setup, running, data origin, working features and planned features.
- Report: problem/track/papers; IR choices and code locations + pipeline diagram; beyond IR; novelty; evaluation; limitations/next steps; work division.
- Work division is informative only: names and actual contributions, no percentages. AI-use declaration has no fixed format; declare uses in report.
- AI agents, APIs, pretrained models, public libraries/data allowed when declared. No direct copying of published projects or prior submissions.
- Partial implementation acceptable; fake/hard-coded/screenshot-only demo gets ZERO working-system marks.
- Frontend not graded; local running is sufficient; deployment not required.
- Credit data and sources. No personal-data collection; obey robots.txt/delays if crawling (we download dataset releases, no crawling).
- Evaluation explicitly asks for a small set judged yourselves OR another justified technique; use public qrels AND provide a human-judgment worksheet.

## Our design decisions (not mandated by the PDF)
- Stay on T1. BM25 + TF-IDF baseline; MiniLM dense; RRF fusion; Streamlit.
- SciFact original dev stance annotations used ONLY as evaluation gold, never runtime evidence decisions.
- No arbitrary weighted truth probability. Transparent support/stability/contradiction gates.
- Separate retrieval relevance from semantic stance; no claim that two IDs prove independence.
