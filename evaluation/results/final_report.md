# Vanilla RAG versus calibrated RAGStress

## 1. Experimental setup

Executed 50 fixed examples, seeded 358, identical corpus/models/hybrid RRF/Top-K=5. Backend hash manifest was frozen before this run. No gold was passed to runtime. Paired system order rotated; caches were cleared before timed runs; model loading excluded.

## 2. Dataset

Original SciFact dev, balanced SUPPORT/CONTRADICT subset. Original test release has no stance labels. This is exploratory: prior prototype work inspected some dev cases. Sentence/document labels describe original claims, not every new generated assertion.

## 3. Vanilla architecture

Hybrid retrieval -> Top-K -> same local FLAN-T5 generation. No runtime claim checking, counter-search, wording probes or trust decisions. Evaluation scoring happens afterward.

## 4. RAGStress architecture

Same retrieval and draft -> sentence-level claim candidates -> independent DeBERTa verification -> corpus counter-search and wording probes -> retained, qualified, corrected or withheld response. Exactly two stress tests.

## 5. Calibration

80 original-training gold rationale pairs. Old correct stance labels across all pairs: 20/80; threshold-only search: 25/80; production safeguards: 23/80. Production decisive coverage: 25/80; conditional accuracy: 23/25. Strong support stays .80; moderate support .60; contradiction .90 plus passage-context confirmation; lexical alignment .15 OR semantic cosine .40; numbered-entity mismatches remain blocking; only Jaccard <.15 materially qualifies strong support. Single supporting documents no longer imply qualification. Decisions are not calibrated truth probabilities. Full old/current gate table and reasons: docs/CALIBRATION.md.

## 6. Methodology

Initial ranked results and deterministic drafts are asserted identical across paired systems. Final factual candidates are evaluated with the same calibrated local NLI and the same initial Top-5 evidence pool. Model-based diagnostics may favor the auditing model and require independent human validation. Retained-claim denominators and answer coverage are disclosed. Support and contradiction rates can overlap for contested claims.

## 7–14. Complete comparison

| Metric | Vanilla | RAGStress | Absolute change | Relative change % |

|---|---:|---:|---:|---:|

| P@5 | 0.1720 | 0.1720 | 0.0000 | 0.0000 |

| Recall@5 | 0.8200 | 0.8200 | 0.0000 | 0.0000 |

| F1@5 | 0.2813 | 0.2813 | 0.0000 | 0.0000 |

| MRR@5 | 0.6817 | 0.6817 | 0.0000 | 0.0000 |

| nDCG@5 | 0.7134 | 0.7134 | 0.0000 | 0.0000 |

| Recall@1 | 0.5800 | 0.5800 | 0.0000 | 0.0000 |

| P@10 | 0.0980 | 0.0980 | 0.0000 | 0.0000 |

| Recall@10 | 0.9150 | 0.9150 | 0.0000 | 0.0000 |

| F1@10 | 0.1752 | 0.1752 | 0.0000 | 0.0000 |

| MRR@10 | 0.6948 | 0.6948 | 0.0000 | 0.0000 |

| nDCG@10 | 0.7450 | 0.7450 | 0.0000 | 0.0000 |

| Claim Support Rate (NLI) | 0.7313 | 0.9677 | 0.2364 | 32.3239 |

| Contradiction Rate (NLI) | 0.2239 | 0.0968 | -0.1271 | -56.7742 |

| Unsupported Claim Rate (NLI) | 0.1194 | 0.0323 | -0.0871 | -72.9839 |

| Evidence Coverage (NLI) | 0.7313 | 0.9677 | 0.2364 | 32.3239 |

| Groundedness (NLI) | 0.6567 | 0.8710 | 0.2143 | 32.6246 |

| Gold evidence sentence recall | 0.5667 | 0.5667 | 0.0000 | 0.0000 |

| Fully supported answer incidence (NLI) | 0.5400 | 0.8000 | 0.2600 | 48.1481 |

| Evidence Precision | Not available — insufficient gold annotation | Not available — insufficient gold annotation | — | — |

| Answer Accuracy | Not available — insufficient gold annotation | Not available — insufficient gold annotation | — | — |

| Answer Precision | Not available — insufficient gold annotation | Not available — insufficient gold annotation | — | — |

| Answer Recall | Not available — insufficient gold annotation | Not available — insufficient gold annotation | — | — |

| Answer F1 | Not available — insufficient gold annotation | Not available — insufficient gold annotation | — | — |

| Selective Accuracy | Not available — insufficient gold annotation | Not available — insufficient gold annotation | — | — |

| Error Rate among answered | Not available — insufficient gold annotation | Not available — insufficient gold annotation | — | — |

| Abstention Rate | 0.0000 | 0.0400 | 0.0400 | — |

| Coverage | 1.0000 | 0.9600 | -0.0400 | -4.0000 |

| Automated answer-stance accuracy | 0.4200 | 0.5200 | 0.1000 | 23.8095 |

| Automated selective stance accuracy | 0.4200 | 0.5417 | 0.1217 | 28.9683 |

| Average Latency (s) | 1.0827 | 11.8335 | 10.7508 | 992.9189 |

| Median Latency (s) | 0.8321 | 9.3963 | 8.5642 | 1029.2755 |

| Counter-Evidence Discovery Rate (NLI) | Not executed | 0.2656 | — | — |

| Query Stability | Not executed | 0.7319 | — | — |

| Automated stance precision | 0.6336 | 0.9048 | 0.2711 | 42.7924 |

| Automated stance recall | 0.4200 | 0.5200 | 0.1000 | 23.8095 |

| Automated stance f1 | 0.4596 | 0.6343 | 0.1747 | 38.0062 |

| Mean retrieval seconds | 0.0539 | 0.0495 | -0.0044 | -8.1286 |

| Mean generation seconds | 1.0289 | 1.0511 | 0.0222 | 2.1585 |

| Mean verification seconds | 0.0000 | 7.4981 | 7.4981 | — |

| Mean counter_evidence seconds | 0.0000 | 3.0542 | 3.0542 | — |

| Mean query_stability seconds | 0.0000 | 0.1641 | 0.1641 | — |

| Mean other_audit seconds | 0.0000 | 0.0166 | 0.0166 | — |

| Gold refuting-source detection rate | 0.4000 | 0.4000 | 0.0000 | 0.0000 |

| RAGStress Overhead (%) | 0.0000 | 992.9189 | 992.9189 | — |

### Metric definitions

P@K: relevant hits/K. Recall@K: retrieved gold documents/all annotated gold documents. F1@K is their per-query harmonic mean, macro averaged. MRR@K uses the first relevant rank; nDCG@K uses binary gold relevance. @10 metrics use supplementary common retrieval outside generation latency. Initial IR scores must be equal by design.

Claim support and evidence coverage: fraction of final factual candidates with SUPPORT or WEAK_SUPPORT under the common NLI judge. Contradiction: fraction with confirmed CONTRADICT. Unsupported: neither label. Groundedness: support without contradiction. These are model diagnostics, not factual truth. Gold sentence recall: annotated original-claim sentences present verbatim in returned passages.

Free-form answer accuracy, evidence precision and primary selective accuracy are unavailable: generated-claim/candidate-passage gold is not exhaustive. Automated answer-stance alignment is separately reported: final factual text as NLI premise, original SciFact claim as hypothesis, scored against its gold SUPPORT/CONTRADICT label; contested outputs are unresolved, withheld outputs abstain. Unresolved and abstained cases are incorrect in all-example diagnostic accuracy. Coverage is answered cases/all cases; diagnostic selective accuracy conditions on answered cases, retaining unresolved errors.

Judged factual candidates: vanilla 67; full 62. Coverage: 1.000 versus 0.960. Do not interpret higher conditional support without this coverage tradeoff.

### Two stress tests

{
  "counter_claims_tested": 64,
  "counter_claims_with_discovery": 17,
  "unqualified_drafts_challenged": 16,
  "unqualified_drafts_challenged_rate": 0.32,
  "pairwise_jaccard_mean": 0.7213359788359789,
  "pairwise_jaccard_median": 0.6666666666666666,
  "pairwise_jaccard_min": 0.1111111111111111,
  "severe_pair_fraction": 0.006666666666666667,
  "severe_query_fraction": 0.0
}

Counter discovery uses relevance/entity/context-confirmed NLI labels, not query wording or human gold for generated claims. Unqualified drafts challenged is an operational potential-false-confidence diagnostic: factual draft without hedge terms and a full-system counter discovery. Its definition partly uses this system, so no independent false-confidence detection accuracy is claimed. Original-gold refuting-source detection is a separate labelled-subset diagnostic requiring matching source ID and contradiction label.

Wording variants preserve the quoted scientific proposition using retrieval wrappers. They are controlled wording probes, not independently human-verified semantic paraphrases. Pairwise Jaccard includes original/variant and variant/variant pairs. Only severe original-to-variant mean instability influences decisions.

## 15. Paired statistics

```json

[
  {
    "metric": "fully_supported_answer",
    "test": "exact McNemar (binomial discordant pairs)",
    "vanilla_only": 0,
    "ragstress_only": 13,
    "statistic": 0,
    "p_value": 0.000244140625,
    "interpretation": "NLI-based diagnostic; not independent human accuracy. Small discordant counts limit power."
  },
  {
    "metric": "gold_stance_agreement",
    "test": "exact McNemar (binomial discordant pairs)",
    "vanilla_only": 6,
    "ragstress_only": 11,
    "statistic": 6,
    "p_value": 0.332305908203125,
    "interpretation": "NLI-based diagnostic; not independent human accuracy. Small discordant counts limit power."
  },
  {
    "metric": "latency",
    "test": "paired bootstrap, 2000 resamples",
    "statistic": 10.750793370008468,
    "ci95": [
      9.070904154064833,
      12.710317600165725
    ],
    "p_value": null,
    "interpretation": "Percentile interval; no bootstrap p-value is invented."
  },
  {
    "metric": "supported_answer_incidence",
    "test": "paired bootstrap, 2000 resamples",
    "statistic": 0.26,
    "ci95": [
      0.14,
      0.38
    ],
    "p_value": null,
    "interpretation": "Percentile interval; no bootstrap p-value is invented."
  }
]

```

Exact McNemar applies only to paired binary model-diagnostic agreement/support events. Bootstrap intervals use paired query resampling. No significance claim is made for unavailable human correctness or for identical IR outcomes.

## 16. Representative error analysis

### both stance-aligned

**823** · gold SUPPORT · N348I mutations cause resistance to zidovudine (AZT).

Vanilla: N348I mutations cause resistance to zidovudine (AZT). [15319019] [2319305] [6426919]

Vanilla sources: 15319019, 2319305, 6426919, 6421792, 641786

RAGStress (KEEP): N348I mutations cause resistance to zidovudine (AZT). [15319019]

Strong relevant support, no confirmed contradiction, and no severe wording instability.

Supporting excerpts: [15319019] N348I appeared early in therapy and was highly associated with thymidine analogue mutations (TAMs) M41L and T215Y/F (p , 0.001), the lamivudine resistance mutations M184V/I (p , 0.001), and non-nucleoside RTI (NNRTI) resistance mutations K103N and Y181C/I (p , 0.001).

Conflicting excerpts:

Counter-discovered claims: 0; wording mean: 0.889

**922** · gold CONTRADICT · Patients in stable partnerships have a faster progression from HIV to AIDS.

Vanilla: A stable partnership is associated with a slower rate of progression to AIDS or death in HIV infected patients receiving HAART. In an analysis stratified by previous antiretroviral therapy and clinical stage when starting HAART (US Centers for Disease Control and Prevention group A, B, or C), the adjusted hazard ratio for progression to AIDS or death was 0.79 (95% confidence interval 0.63 to 0.98) for participants with a stable partnership [17077004] [11230569] [31562330]

Vanilla sources: 17077004, 11230569, 31562330, 46353045, 374902

RAGStress (KEEP): A stable partnership is associated with a slower rate of progression to AIDS or death in HIV infected patients receiving HAART. [17077004] [24596228]

In an analysis stratified by previous antiretroviral therapy and clinical stage when starting HAART (US Centers for Disease Control and Prevention group A, B, or C), the adjusted hazard ratio for progression to AIDS or death was 0.79 (95% confidence interval 0.63 to 0.98) for participants with a stable partnership [17077004]

Strong relevant support, no confirmed contradiction, and no severe wording instability.; Strong relevant support, no confirmed contradiction, and no severe wording instability.

Supporting excerpts: [17077004] CONCLUSIONS A stable partnership is associated with a slower rate of progression to AIDS or death in HIV infected patients receiving HAART. | [17077004] In an analysis stratified by previous antiretroviral therapy and clinical stage when starting HAART (US Centers for Disease Control and Prevention group A, B, or C), the adjusted hazard ratio for progression to AIDS or death was 0.79 (95% confidence interval 0.63 to 0.98) for participants with a stable partnership compared with those without.

Conflicting excerpts:

Counter-discovered claims: 0; wording mean: 0.667

### vanilla misaligned/full aligned

**674** · gold CONTRADICT · LDL cholesterol has no involvement in the development of cardiovascular disease.

Vanilla: LDL cholesterol has no involvement in the development of cardiovascular disease. [2095573] [7662206] [9814332]

Vanilla sources: 2095573, 7662206, 9814332, 15360986, 30981192

RAGStress (REWRITE): The proposed claim is not retained. Source [30981192] reports: "Lowering low-density lipoprotein-cholesterol (LDL-C) is the primary target in the management of dyslipidemia in patients at high risk of cardiovascular disease."

Relevant corpus evidence contradicts the wording; replace it with an attributed passage.

Supporting excerpts:

Conflicting excerpts: [2095573] These results potentially provide insight into the biological mechanisms that underlie the regulation of LDL cholesterol and might help in the discovery of novel therapeutic targets for cardiovascular disease. | [2095573] SNPs rs599839 and rs646776 both explained around 1% of the variation in circulating LDL-cholesterol concentrations and were associated with about 15% of an SD change in LDL cholesterol per allele, assuming an SD of 1 mmol/L.

Counter-discovered claims: 1; wording mean: 0.889

**692** · gold CONTRADICT · Leuko-increased blood increases infectious complications in red blood cell transfusion.

Vanilla: Leuko-increased blood increases infectious complications in red blood cell transfusion. [24088502] [3698758] [11349166]

Vanilla sources: 24088502, 3698758, 11349166, 23203102, 623486

RAGStress (REWRITE): The proposed claim is not retained. Source [11349166] reports: "RESULTS The study included a total of 36 blood donors who subsequently developed CJD and 436 recipients. Through 2006, 91 of these recipients were still alive, 329 were deceased, and 16 were lost to follow-up. After transfusion, these three groups had survived a total of 2096.0 person-years. A total of 144 recipients survived 5 years or longer after transfusion and 68 of them had received blood donated 60 or fewer months before the onset of CJD in the donor. We identified no recipient with CJD. CONCLUSIONS The current results of this large, ongoing lookback study show no evidence of transfusion transmission of CJD. They reinforce the conclusion that the risk, if any, of transfusion transmission of prion disease by CJD donors is significantly lower than the comparable risk of such transmission by vCJD donors."

Relevant corpus evidence contradicts the wording; replace it with an attributed passage.

Supporting excerpts:

Conflicting excerpts: [24088502] CONCLUSION A national universal leukoreduction program is potentially associated with decreased mortality as well as decreased fever episodes and antibiotic use after red blood cell transfusion in high-risk patients. | [24088502] The frequency of posttransfusion fevers decreased significantly following leukoreduction (adjusted OR, 0.86; 95% CI, 0.79-0.94), as did antibiotic use (adjusted OR, 0.90; 95% CI, 0.82-0.99).

Counter-discovered claims: 1; wording mean: 0.667

### unqualified draft challenged

**674** · gold CONTRADICT · LDL cholesterol has no involvement in the development of cardiovascular disease.

Vanilla: LDL cholesterol has no involvement in the development of cardiovascular disease. [2095573] [7662206] [9814332]

Vanilla sources: 2095573, 7662206, 9814332, 15360986, 30981192

RAGStress (REWRITE): The proposed claim is not retained. Source [30981192] reports: "Lowering low-density lipoprotein-cholesterol (LDL-C) is the primary target in the management of dyslipidemia in patients at high risk of cardiovascular disease."

Relevant corpus evidence contradicts the wording; replace it with an attributed passage.

Supporting excerpts:

Conflicting excerpts: [2095573] These results potentially provide insight into the biological mechanisms that underlie the regulation of LDL cholesterol and might help in the discovery of novel therapeutic targets for cardiovascular disease. | [2095573] SNPs rs599839 and rs646776 both explained around 1% of the variation in circulating LDL-cholesterol concentrations and were associated with about 15% of an SD change in LDL cholesterol per allele, assuming an SD of 1 mmol/L.

Counter-discovered claims: 1; wording mean: 0.889

**692** · gold CONTRADICT · Leuko-increased blood increases infectious complications in red blood cell transfusion.

Vanilla: Leuko-increased blood increases infectious complications in red blood cell transfusion. [24088502] [3698758] [11349166]

Vanilla sources: 24088502, 3698758, 11349166, 23203102, 623486

RAGStress (REWRITE): The proposed claim is not retained. Source [11349166] reports: "RESULTS The study included a total of 36 blood donors who subsequently developed CJD and 436 recipients. Through 2006, 91 of these recipients were still alive, 329 were deceased, and 16 were lost to follow-up. After transfusion, these three groups had survived a total of 2096.0 person-years. A total of 144 recipients survived 5 years or longer after transfusion and 68 of them had received blood donated 60 or fewer months before the onset of CJD in the donor. We identified no recipient with CJD. CONCLUSIONS The current results of this large, ongoing lookback study show no evidence of transfusion transmission of CJD. They reinforce the conclusion that the risk, if any, of transfusion transmission of prion disease by CJD donors is significantly lower than the comparable risk of such transmission by vCJD donors."

Relevant corpus evidence contradicts the wording; replace it with an attributed passage.

Supporting excerpts:

Conflicting excerpts: [24088502] CONCLUSION A national universal leukoreduction program is potentially associated with decreased mortality as well as decreased fever episodes and antibiotic use after red blood cell transfusion in high-risk patients. | [24088502] The frequency of posttransfusion fevers decreased significantly following leukoreduction (adjusted OR, 0.86; 95% CI, 0.79-0.94), as did antibiotic use (adjusted OR, 0.90; 95% CI, 0.82-0.99).

Counter-discovered claims: 1; wording mean: 0.667

### unsupported draft qualified

No observed example in this fixed run; none fabricated.

### refuting source additionally detected

No observed example in this fixed run; none fabricated.

### both stance-misaligned

**217** · gold SUPPORT · CX3CR1 on the Th2 cells promotes T cell survival

Vanilla: CX3CR1-induced survival was also observed for T(H)1 cells upon airway inflammation but not under homeostatic conditions or upon peripheral inflammation. Therefore, CX3CR1 and CX3CL1 may represent attractive therapeutic targets in asthma. [21366394] [21366394] [21363424] [22210434]

Vanilla sources: 21366394, 21363424, 22210434, 41877386, 45414636

RAGStress (KEEP): CX3CR1-induced survival was also observed for T(H)1 cells upon airway inflammation but not under homeostatic conditions or upon peripheral inflammation. [21366394]

Therefore, CX3CR1 and CX3CL1 may represent attractive therapeutic targets in asthma. [21366394]

Strong relevant support, no confirmed contradiction, and no severe wording instability.; Strong relevant support, no confirmed contradiction, and no severe wording instability.

Supporting excerpts: [21366394] CX3CR1-induced survival was also observed for T(H)1 cells upon airway inflammation but not under homeostatic conditions or upon peripheral inflammation. | [21366394] Therefore, CX3CR1 and CX3CL1 may represent attractive therapeutic targets in asthma.

Conflicting excerpts:

Counter-discovered claims: 0; wording mean: 0.587

**1368** · gold SUPPORT · Vitamin D deficiency effects the term of delivery.

Vanilla: Vitamin D deficiency and vitamin D deficiency are common in elderly people, especially in those who are infirm and not exposed to sunlight or who live at latitudes that do not provide them with sunlight-mediated cholecalciferol during the winter months. [23267371] [275294] [30720103]

Vanilla sources: 23267371, 275294, 30720103, 3150030, 21553394

RAGStress (KEEP): Vitamin D deficiency and vitamin D deficiency are common in elderly people, especially in those who are infirm and not exposed to sunlight or who live at latitudes that do not provide them with sunlight-mediated cholecalciferol during the winter months. [275294]

Strong relevant support, no confirmed contradiction, and no severe wording instability.

Supporting excerpts: [275294] It is now recognized that vitamin D insufficiency and vitamin D deficiency are common in elderly people, especially in those who are infirm and not exposed to sunlight or who live at latitudes that do not provide them with sunlight-mediated cholecalciferol during the winter months.

Conflicting excerpts:

Counter-discovered claims: 0; wording mean: 1.000

### supported draft withheld or qualified

**233** · gold CONTRADICT · Cell autonomous sex determination in somatic cells does not occur in Galliformes.

Vanilla: Our study demonstrates that avian somatic cells possess an inherent sex identity and that, in birds, sexual differentiation is substantively cell autonomous. In contrast to the outcome for mammalian mixed-sex chimaeras, in chicken mixed-sex chimaeras the donor cells were excluded from the functional structures of the host gonad. [36708463] A major question is whether genes encoded on the sex [4388470] [36708463] [28390999]

Vanilla sources: 4388470, 36708463, 28390999, 12232678, 23581096

RAGStress (QUALIFY): Corpus evidence is contested for: Our study demonstrates that avian somatic cells possess an inherent sex identity and that, in birds, sexual differentiation is substantively cell autonomous. Supporting: [4388470]; conflicting: [41380943]

Corpus evidence is contested for: In contrast to the outcome for mammalian mixed-sex chimaeras, in chicken mixed-sex chimaeras the donor cells were excluded from the functional structures of the host gonad. Supporting: [4388470]; conflicting: [41380943]

A major question is whether genes encoded on the sex [36708463]

Relevant passages support and contradict the claim.; Relevant passages support and contradict the claim.; Strong relevant support, no confirmed contradiction, and no severe wording instability.

Supporting excerpts: [4388470] Our study demonstrates that avian somatic cells possess an inherent sex identity and that, in birds, sexual differentiation is substantively cell autonomous. | [4388470] In contrast to the outcome for mammalian mixed-sex chimaeras, in chicken mixed-sex chimaeras the donor cells were excluded from the functional structures of the host gonad. In an example where female tissue was transplanted into a male host, donor cells contributing to the developing testis retained a female identity and expressed a marker of female function. Our study demonstrates that avian somatic cells possess an inherent sex identity and that, in birds, sexual differentiation is substantively cell autonomous.

Conflicting excerpts: [41380943] During embryonic development, gonadal steroid hormones (androgens and estrogens) are thought to organize the sexual differentiation of the brain in the heterogametic sexes of higher vertebrates (males in mammals, females in birds). | [41380943] During embryonic development, gonadal steroid hormones (androgens and estrogens) are thought to organize the sexual differentiation of the brain in the heterogametic sexes of higher vertebrates (males in mammals, females in birds). Brain differentiation of the homogametic sexes is thought to proceed by default, not requiring sex hormones for sex-specific organization. In gallinaceous birds such as the Japanese quail, female brain organization is thought to develop via estrogen-dependent demasculinization of a default male brain phenotype. We performed male donor-to-female host (MF), female-to-male (FM), male-to-male (MM), and female-to-female (FF) isotopic, isochronic transplantation of the forebrain primordium in Japanese quail embryos before gonadal differentiation had occurred; brain chimeras had a forebrain (including the hypothalamus) originating exclusively from donor cells. MM, FF, and MF chimeras all showed sexual behavior governed by the genetic sex of the host.

Counter-discovered claims: 2; wording mean: 0.667

### Previous false rejection examples (training diagnosis)

IL-10 production by monocytes inhibits CD4 + T cell response.

Triggering of PD-1 expressed on monocytes by PD-L1 expressed on various cell types induced IL-10 production and led to reversible CD4+ T cell dysfunction.

Old: NEUTRAL; new: WEAK_SUPPORT; support=0.719, contradiction=0.153, coverage=0.783, cosine=0.775. See stored entity/context diagnostics; no dev-based retuning.

Medications to treat obesity have unwanted side effects.

Orlistat reduced the incidence of diabetes and improved concentrations of total cholesterol and low density lipoprotein cholesterol, blood pressure, and glycaemic control in patients with diabetes but increased rates of gastrointestinal side effects and slightly lowered concentrations of high density lipoprotein.

Old: NEUTRAL; new: SUPPORT; support=0.928, contradiction=0.010, coverage=0.234, cosine=0.374. See stored entity/context diagnostics; no dev-based retuning.

Klf2 is important for proper myeloid cell function.

Herein, we identify the Kruppel-like transcription factor 2 (KLF2) as a potent regulator of myeloid cell activation in vivo.

Old: NEUTRAL; new: WEAK_SUPPORT; support=0.615, contradiction=0.000, coverage=0.557, cosine=0.839. See stored entity/context diagnostics; no dev-based retuning.

## 17. Component comparison

Vanilla / NLI-only / full auditing, all actually executed. NLI-only includes the existing claim-based retrieval and quote-recovery path, but neither counter-search nor wording probes. Therefore this isolates that verification layer, not NLI inference alone.

| System | Coverage | NLI support | NLI groundedness | Automated stance agreement | Mean seconds |

|---|---:|---:|---:|---:|---:|

| vanilla | 1.0000 | 0.7313 | 0.6567 | 0.4200 | 1.0827 |

| nli_only | 0.9600 | 0.9677 | 0.8710 | 0.5200 | 8.9831 |

| ragstress | 0.9600 | 0.9677 | 0.8710 | 0.5200 | 11.8335 |

Actual full-system decision distribution: {"KEEP": 26, "REWRITE": 16, "ABSTAIN": 2, "QUALIFY": 6}

### Incremental contribution of the stress tests

In this executed subset, NLI-only and full auditing have identical aggregate claim support, groundedness, answer coverage, fully-supported answer incidence and overall automated stance agreement. Full auditing is slower. Therefore the results do not establish an incremental answer-quality benefit from counter-search or wording probes beyond the existing verification/recovery layer. The probes still provide inspectable evidence diagnostics. No query crosses the severe mean-overlap threshold; a few individual pairs do.

### Directional failures in the stance diagnostic

The automated confusion matrices show gold-SUPPORT agreement 15/25 versus 9/25, and gold-CONTRADICT agreement 6/25 versus 17/25. These class-specific outcomes must not be hidden by the overall diagnostic. The same NLI judge infers answer relation, so these are model-based failure signals requiring human review.

## 18. Limitations

Small balanced exploratory subset does not represent natural label prevalence; prior dev exposure; generic NLI scientific-domain errors; same-model auditing/scoring circularity; no independent generated-claim gold; heuristic sentence extraction and alias rules; incomplete literature/corpus annotations; wording-probe equivalence not human-reviewed; extra retrieval budget for counter-search; warm CPU timing and quote recovery confounds. Verbatim quotation recovery naturally improves text-to-evidence entailment without establishing independent scientific truth. Human review is still required.

## 19. Measured conclusion

Initial IR is identical, as required. Model-based groundedness changes from 0.657 to 0.871; answer coverage from 1.000 to 0.960. Mean latency changes from 1.083s to 11.834s (992.9% overhead). Exact McNemar p-values: fully supported answer incidence 0.000244141; automated stance agreement 0.332305908. Interpret these with their stated diagnostic limitations. NLI-only matches the main aggregate quality metrics at lower cost; stress-test-specific gains are not established here. These results describe this corpus and automated assessment, not universal trustworthiness or verified free-form answer correctness.

## Reproduction

Run `python run_evaluation.py`. Verified matching checkpoints resume the fixed experiment; use `--no-resume` for an entirely new timed rerun. Freeze checks prevent silently changing the runtime during this study. Raw results and eight figures are included.

## Implementation handoff

Calibration details, old/current gates and reasons: docs/CALIBRATION.md. Created: src/vanilla.py, evaluation/study.py, evaluation/analysis.py, evaluation/evaluation_dataset.json, run_evaluation.py, calibration artifacts, tests/test_calibrated_policy.py and tests/test_study.py. Modified: src/config.py, src/stance.py, src/trust.py, src/pipeline.py, src/evaluation.py, src/cli.py, app.py, src/presentation.py, important IR/presentation/answer-repair tests, README and design/experiment/UI/AI-use/video documentation. The obsolete dependency stage and its historical active outputs were deleted. An archive is outside the active submission. Backend sanity runs and 34 tests passed; UI rendering checks use recorded real outputs and are not presented as a live performance experiment.

## Computational cost

All compared runs use local CPU models; no remote API is called. Latency measures computational overhead, not electricity or dollar cost. Model loading/indexing and post-hoc evaluation are outside timed answering. Stage-wise timing appears in the comparison table.
