import csv,json,math
from collections import Counter
import numpy as np
from scipy.stats import binomtest
from .study import OUT,ROOT,NA,save,gold_recall

def mean(values):return float(np.mean(values)) if values else None
def ratio(a,b):return a/b if b else None

def classification(records,name):
    labels=['SUPPORT','CONTRADICT'];preds=labels+['UNRESOLVED','ABSTAIN']
    matrix=[[0]*4 for _ in labels]
    for r in records:
        actual=r['example']['gold_label'];pred=r['systems'][name]['evaluation']['answer_stance_diagnostic']['predicted']
        matrix[labels.index(actual)][preds.index(pred)]+=1
    metrics=[]
    for i in range(2):
        tp=matrix[i][i];fp=matrix[1-i][i];fn=sum(matrix[i])-tp
        p=tp/(tp+fp) if tp+fp else 0;r=tp/(tp+fn) if tp+fn else 0
        metrics.append({'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0})
    return {'labels':labels,'predictions':preds,'matrix':matrix,
        **{key:mean([m[key] for m in metrics]) for key in ['precision','recall','f1']}}

def summarize(records,name):
    runs=[r['systems'][name] for r in records];evaluated=[r['evaluation'] for r in runs]
    claims=[c for e in evaluated for c in e['claim_judgments']]
    answered=[e for e in evaluated if e['answered']]
    summary={k:mean([e['ir'][k] for e in evaluated]) for k in evaluated[0]['ir']}
    summary.update({'examples':len(runs),'judged_claims':len(claims),
        'Claim Support Rate (NLI)':ratio(sum(c['supported'] for c in claims),len(claims)),
        'Contradiction Rate (NLI)':ratio(sum(c['contradicted'] for c in claims),len(claims)),
        'Unsupported Claim Rate (NLI)':ratio(sum(c['unsupported'] for c in claims),len(claims)),
        'Evidence Coverage (NLI)':ratio(sum(c['supported'] for c in claims),len(claims)),
        'Groundedness (NLI)':ratio(sum(c['supported'] and not c['contradicted'] for c in claims),len(claims)),
        'Gold evidence sentence recall':mean([e['gold_sentence_recall'] for e in evaluated if e['gold_sentence_recall'] is not None]),
        'Fully supported answer incidence (NLI)':mean([e['fully_supported_answer'] for e in evaluated]),
        'Evidence Precision':NA,'Answer Accuracy':NA,'Answer Precision':NA,'Answer Recall':NA,'Answer F1':NA,
        'Selective Accuracy':NA,'Error Rate among answered':NA,
        'Abstention Rate':1-len(answered)/len(runs),'Coverage':len(answered)/len(runs),
        'Automated answer-stance accuracy':mean([e['gold_stance_agreement'] for e in evaluated]),
        'Automated selective stance accuracy':mean([e['gold_stance_agreement'] for e in answered]),
        'Average Latency (s)':mean([r['elapsed_seconds'] for r in runs]),
        'Median Latency (s)':float(np.median([r['elapsed_seconds'] for r in runs])),
        'Counter-Evidence Discovery Rate (NLI)':None,'Query Stability':None})
    diagnostic=classification(records,name)
    summary.update({'Automated stance '+k:diagnostic[k] for k in ['precision','recall','f1']})
    for key in ['retrieval','generation','verification','counter_evidence','query_stability','other_audit']:
        summary[f'Mean {key} seconds']=mean([r['timings'].get(key,0.0) for r in runs])
    if name=='ragstress':
        summary['Counter-Evidence Discovery Rate (NLI)']=ratio(sum(r['counter_discovery_claims'] for r in records),sum(r['counter_tested_claims'] for r in records))
        summary['Query Stability']=mean([r['stability']['mean_jaccard'] for r in runs])
        summary['decisions']=dict(Counter(r['decision'] for r in runs))
    elif name=='nli_only':summary['decisions']=dict(Counter(r['decision'] for r in runs))
    refutes=[r for r in records if r['gold_refutation_diagnostic']['eligible']]
    if name in ['vanilla','ragstress']:
        summary['Gold refuting-source detection rate']=ratio(sum(r['gold_refutation_diagnostic'][name] for r in refutes),len(refutes))
    return summary,diagnostic

def paired_stats(records):
    rng=np.random.default_rng(358);result=[]
    for field in ['fully_supported_answer','gold_stance_agreement']:
        pairs=[(r['systems']['vanilla']['evaluation'][field],r['systems']['ragstress']['evaluation'][field]) for r in records]
        b=sum(a and not z for a,z in pairs);c=sum(z and not a for a,z in pairs);n=b+c
        result.append({'metric':field,'test':'exact McNemar (binomial discordant pairs)','vanilla_only':b,'ragstress_only':c,
                       'statistic':min(b,c),'p_value':float(binomtest(b,n,.5).pvalue) if n else 1.0,
                       'interpretation':'NLI-based diagnostic; not independent human accuracy. Small discordant counts limit power.'})
    for metric in ['latency','supported_answer_incidence']:
        if metric=='latency':diffs=np.array([r['systems']['ragstress']['elapsed_seconds']-r['systems']['vanilla']['elapsed_seconds'] for r in records])
        else:diffs=np.array([int(r['systems']['ragstress']['evaluation']['fully_supported_answer'])-int(r['systems']['vanilla']['evaluation']['fully_supported_answer']) for r in records])
        means=diffs[rng.integers(0,len(diffs),(2000,len(diffs)))].mean(axis=1)
        result.append({'metric':metric,'test':'paired bootstrap, 2000 resamples','statistic':float(diffs.mean()),
            'ci95':list(map(float,np.quantile(means,[.025,.975]))),'p_value':None,
            'interpretation':'Percentile interval; no bootstrap p-value is invented.'})
    return result

def error_examples(records):
    groups={name:[] for name in ['both stance-aligned','vanilla misaligned/full aligned','unqualified draft challenged',
        'unsupported draft qualified','refuting source additionally detected','both stance-misaligned','supported draft withheld or qualified']}
    for r in records:
        v=r['systems']['vanilla'];f=r['systems']['ragstress'];ve=v['evaluation'];fe=f['evaluation']
        conditions=[ve['gold_stance_agreement'] and fe['gold_stance_agreement'],
            not ve['gold_stance_agreement'] and fe['gold_stance_agreement'],r['potential_false_confidence'],
            any(c['unsupported'] for c in ve['claim_judgments']) and f['decision']=='QUALIFY',
            r['gold_refutation_diagnostic']['eligible'] and not r['gold_refutation_diagnostic']['vanilla'] and r['gold_refutation_diagnostic']['ragstress'],
            not ve['gold_stance_agreement'] and not fe['gold_stance_agreement'],
            ve['fully_supported_answer'] and f['decision'] in ['QUALIFY','ABSTAIN']]
        for key,condition in zip(groups,conditions):
            if condition and len(groups[key])<2:
                evidence=[e for c in f['claims'] for e in c['evidence']]
                groups[key].append({'id':r['example']['id'],'query':r['example']['query'],'gold_label':r['example']['gold_label'],
                    'vanilla_answer':v['generated_answer'],'vanilla_sources':[h['source_id'] for h in v['retrieved']],
                    'ragstress_answer':f['final_answer'],'decision':f['decision'],
                    'supporting_evidence':[e for e in evidence if e['label'] in ['SUPPORT','WEAK_SUPPORT']][:2],
                    'conflicting_evidence':[e for e in evidence if e['label']=='CONTRADICT'][:2],
                    'counter_discovery_claims':r['counter_discovery_claims'],'query_stability':f['stability'],
                    'explanation':'; '.join(c['reason'] for c in f['claims'])})
    return groups

def plots(summaries,diagnostics,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=OUT/'plots';directory.mkdir(exist_ok=True)
    names=['vanilla','ragstress'];colors=['#527da5','#499982']
    def bars(keys,file,title):
        fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(keys));width=.34
        for i,name in enumerate(names):ax.bar(x+(i-.5)*width,[summaries[name][k] or 0 for k in keys],width,label=name,color=colors[i])
        ax.set_xticks(x,keys,rotation=15,ha='right');ax.set_title(title);ax.legend();fig.tight_layout();fig.savefig(directory/file,dpi=180);plt.close(fig)
    bars(['P@5','Recall@5','F1@5','MRR@5','nDCG@5'],'ir.png','Same initial hybrid retrieval in both systems')
    bars(['Claim Support Rate (NLI)','Unsupported Claim Rate (NLI)','Contradiction Rate (NLI)'],'claims.png','Model-based claim assessment; rates can overlap')
    bars(['Automated answer-stance accuracy','Automated selective stance accuracy'],'stance.png','Automated stance diagnostic, not free-form correctness')
    bars(['Average Latency (s)','Median Latency (s)'],'latency.png','Warm timed executions; model loading excluded')
    fig,ax=plt.subplots(figsize=(8,4));ax.hist([r['systems']['ragstress']['stability']['mean_jaccard'] for r in records],bins=10,color=colors[1]);ax.axvline(.15,color='red',ls='--',label='severe threshold');ax.set(title='Wording-probe source-set overlap',xlabel='Mean original-to-variant Jaccard',ylabel='Queries');ax.legend();fig.tight_layout();fig.savefig(directory/'stability.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(['Detected','Not detected'],[sum(r['counter_discovery_claims'] for r in records),sum(r['counter_tested_claims']-r['counter_discovery_claims'] for r in records)],color=colors);ax.set(title='Counter-evidence discovery: calibrated NLI diagnostic',ylabel='Audited claims');fig.tight_layout();fig.savefig(directory/'counter.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,name in zip(axes,names):
        d=diagnostics[name];matrix=np.array(d['matrix']);ax.imshow(matrix,cmap='Blues');ax.set_xticks(range(4),d['predictions'],rotation=25);ax.set_yticks(range(2),d['labels']);ax.set_title(name+' · automated answer-stance');ax.set_xlabel('NLI inferred output relation');ax.set_ylabel('SciFact original-claim gold')
        for i in range(2):
            for j in range(4):ax.text(j,i,str(matrix[i,j]),ha='center',va='center')
    fig.tight_layout();fig.savefig(directory/'confusion.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));labels=['KEEP','QUALIFY','REWRITE','ABSTAIN'];ax.bar(labels,[summaries['ragstress']['decisions'].get(l,0) for l in labels],color=['#499982','#d7a85d','#527da5','#aa6464']);ax.set(title='Actual full-system decisions',ylabel='Examples');fig.tight_layout();fig.savefig(directory/'decisions.png',dpi=180);plt.close(fig)

def analyze(records,protocol):
    # Repair the scope of this post-hoc diagnostic in older checkpoints; no answering or timing is rerun.
    corrected = 0
    for index, record in enumerate(records):
        diagnostic = record['gold_refutation_diagnostic']
        if 'counter_only' not in diagnostic:
            diagnostic['counter_only'] = diagnostic['ragstress']
            diagnostic['ragstress'] = (diagnostic['vanilla'] or diagnostic['counter_only']) if diagnostic['eligible'] else None
            full = record['systems']['ragstress']
            counter_hits = [h for c in full['claims'] for run in c['counter_evidence']['runs'] for h in run['hits']]
            diagnostic['expanded_gold_sentence_recall'] = gold_recall(full['retrieved'] + counter_hits, record['example'])
            save(OUT/'raw'/f"{index:03d}-{record['example']['id']}.json",record)
            corrected += 1
    protocol['refuting_evidence_scope'] = 'Initial Top-5 union counter-search evidence; counter-only result stored separately.'
    if corrected:
        protocol['posthoc_reporting_correction'] = 'Corrected counter-only versus union diagnostic scope; production outputs and measured timings unchanged.'
    save(OUT/'raw_results.json',records);save(OUT/'protocol.json',protocol)
    summaries={};diagnostics={}
    for name in ['vanilla','nli_only','ragstress']:summaries[name],diagnostics[name]=summarize(records,name)
    v,f=summaries['vanilla'],summaries['ragstress']
    f['RAGStress Overhead (%)']=(f['Average Latency (s)']-v['Average Latency (s)'])/v['Average Latency (s)']*100
    rows=[]
    for key in v:
        if key in ['decisions','examples','judged_claims']:continue
        a,b=v[key],f.get(key);numeric=isinstance(a,(float,int)) and isinstance(b,(float,int))
        rows.append({'Metric':key,'Vanilla RAG':a if a is not None else 'Not executed',
            'RAGStress':b if b is not None else 'Not executed','Absolute Change':b-a if numeric else None,
            'Relative Change (%)':(b-a)/a*100 if numeric and a else None})
    rows.append({'Metric':'RAGStress Overhead (%)','Vanilla RAG':0,'RAGStress':f['RAGStress Overhead (%)'],'Absolute Change':f['RAGStress Overhead (%)'],'Relative Change (%)':None})
    save(OUT/'summary.json',summaries);save(OUT/'comparison.json',rows);save(OUT/'confusion_matrices.json',diagnostics)
    with (OUT/'comparison.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    stats=paired_stats(records);save(OUT/'statistics.json',stats)
    errors=error_examples(records);save(OUT/'error_analysis.json',errors)
    pairwise=[v for r in records for v in r['stability_pairwise']]
    stress={'counter_claims_tested':sum(r['counter_tested_claims'] for r in records),
        'counter_claims_with_discovery':sum(r['counter_discovery_claims'] for r in records),
        'unqualified_drafts_challenged':sum(r['potential_false_confidence'] for r in records),
        'unqualified_drafts_challenged_rate':mean([r['potential_false_confidence'] for r in records]),
        'pairwise_jaccard_mean':mean(pairwise),'pairwise_jaccard_median':float(np.median(pairwise)),
        'pairwise_jaccard_min':min(pairwise),'severe_pair_fraction':mean([p<.15 for p in pairwise]),
        'severe_query_fraction':mean([r['systems']['ragstress']['stability']['mean_jaccard']<.15 for r in records])}
    save(OUT/'stress_metrics.json',stress);plots(summaries,diagnostics,records)
    report(rows,summaries,stats,errors,stress,protocol,diagnostics)
    return summaries

def report(comparison,summaries,stats,errors,stress,protocol,diagnostics):
    def fmt(value):return f'{value:.4f}' if isinstance(value,(float,int)) else str(value if value is not None else '—')
    support_p=next(s['p_value'] for s in stats if s['metric']=='fully_supported_answer')
    stance_p=next(s['p_value'] for s in stats if s['metric']=='gold_stance_agreement')
    def agreed(name,index):
        row=diagnostics[name]['matrix'][index]
        return f'{row[index]}/{sum(row)}'
    text=['# Vanilla RAG versus calibrated RAGStress',
      '## 1. Experimental setup',f"Executed {protocol['examples']} fixed examples, seeded {protocol['seed']}, identical corpus/models/hybrid RRF/Top-K=5. Backend hash manifest was frozen before this run. No gold was passed to runtime. Paired system order rotated; caches were cleared before timed runs; model loading excluded.",
      '## 2. Dataset', 'Original SciFact dev, balanced SUPPORT/CONTRADICT subset. Original test release has no stance labels. This is exploratory: prior prototype work inspected some dev cases. Sentence/document labels describe original claims, not every new generated assertion.',
      '## 3. Vanilla architecture','Hybrid retrieval -> Top-K -> same local FLAN-T5 generation. No runtime claim checking, counter-search, wording probes or trust decisions. Evaluation scoring happens afterward.',
      '## 4. RAGStress architecture','Same retrieval and draft -> sentence-level claim candidates -> independent DeBERTa verification -> corpus counter-search and wording probes -> retained, qualified, corrected or withheld response. Exactly two stress tests.',
      '## 5. Calibration','80 original-training gold rationale pairs. Old correct stance labels across all pairs: 20/80; threshold-only search: 25/80; production safeguards: 23/80. Production decisive coverage: 25/80; conditional accuracy: 23/25. Strong support stays .80; moderate support .60; contradiction .90 plus passage-context confirmation; lexical alignment .15 OR semantic cosine .40; numbered-entity mismatches remain blocking; only Jaccard <.15 materially qualifies strong support. Single supporting documents no longer imply qualification. Decisions are not calibrated truth probabilities. Full old/current gate table and reasons: docs/CALIBRATION.md.',
      '## 6. Methodology','Initial ranked results and deterministic drafts are asserted identical across paired systems. Final factual candidates are evaluated with the same calibrated local NLI and the same initial Top-5 evidence pool. Model-based diagnostics may favor the auditing model and require independent human validation. Retained-claim denominators and answer coverage are disclosed. Support and contradiction rates can overlap for contested claims.',
      '## 7–14. Complete comparison',
      '| Metric | Vanilla | RAGStress | Absolute change | Relative change % |','|---|---:|---:|---:|---:|']
    text.extend('| '+' | '.join(fmt(r[k]) for k in ['Metric','Vanilla RAG','RAGStress','Absolute Change','Relative Change (%)'])+' |' for r in comparison)
    text.extend(['### Metric definitions',
      'P@K: relevant hits/K. Recall@K: retrieved gold documents/all annotated gold documents. F1@K is their per-query harmonic mean, macro averaged. MRR@K uses the first relevant rank; nDCG@K uses binary gold relevance. @10 metrics use supplementary common retrieval outside generation latency. Initial IR scores must be equal by design.',
      'Claim support and evidence coverage: fraction of final factual candidates with SUPPORT or WEAK_SUPPORT under the common NLI judge. Contradiction: fraction with confirmed CONTRADICT. Unsupported: neither label. Groundedness: support without contradiction. These are model diagnostics, not factual truth. Gold sentence recall: annotated original-claim sentences present verbatim in returned passages.',
      'Free-form answer accuracy, evidence precision and primary selective accuracy are unavailable: generated-claim/candidate-passage gold is not exhaustive. Automated answer-stance alignment is separately reported: final factual text as NLI premise, original SciFact claim as hypothesis, scored against its gold SUPPORT/CONTRADICT label; contested outputs are unresolved, withheld outputs abstain. Unresolved and abstained cases are incorrect in all-example diagnostic accuracy. Coverage is answered cases/all cases; diagnostic selective accuracy conditions on answered cases, retaining unresolved errors.',
      f"Judged factual candidates: vanilla {summaries['vanilla']['judged_claims']}; full {summaries['ragstress']['judged_claims']}. Coverage: {summaries['vanilla']['Coverage']:.3f} versus {summaries['ragstress']['Coverage']:.3f}. Do not interpret higher conditional support without this coverage tradeoff.",
      '### Two stress tests',json.dumps(stress,indent=2),
      'Counter discovery uses relevance/entity/context-confirmed NLI labels, not query wording or human gold for generated claims. Unqualified drafts challenged is an operational potential-false-confidence diagnostic: factual draft without hedge terms and a full-system counter discovery. Its definition partly uses this system, so no independent false-confidence detection accuracy is claimed. Original-gold refuting-source detection is a separate labelled-subset diagnostic requiring matching source ID and contradiction label.',
      'Wording variants preserve the quoted scientific proposition using retrieval wrappers. They are controlled wording probes, not independently human-verified semantic paraphrases. Pairwise Jaccard includes original/variant and variant/variant pairs. Only severe original-to-variant mean instability influences decisions.',
      '## 15. Paired statistics','```json',json.dumps(stats,indent=2),'```',
      'Exact McNemar applies only to paired binary model-diagnostic agreement/support events. Bootstrap intervals use paired query resampling. No significance claim is made for unavailable human correctness or for identical IR outcomes.',
      '## 16. Representative error analysis'])
    for category,examples in errors.items():
        text.append('### '+category)
        if not examples:text.append('No observed example in this fixed run; none fabricated.')
        for e in examples:
            text.extend([f"**{e['id']}** · gold {e['gold_label']} · {e['query']}",
                'Vanilla: '+e['vanilla_answer'],'Vanilla sources: '+', '.join(e['vanilla_sources']),
                f"RAGStress ({e['decision']}): {e['ragstress_answer']}",e['explanation'],
                'Supporting excerpts: '+' | '.join(f"[{r['source_id']}] {r['quote']}" for r in e['supporting_evidence']),
                'Conflicting excerpts: '+' | '.join(f"[{r['source_id']}] {r['quote']}" for r in e['conflicting_evidence']),
                f"Counter-discovered claims: {e['counter_discovery_claims']}; wording mean: {e['query_stability']['mean_jaccard']:.3f}"])
    train=json.loads((ROOT/'calibration/old_new_training_diagnosis.json').read_text(encoding='utf8'))
    changed=[r for r in train if r['gold']=='SUPPORT' and r['old_label']=='NEUTRAL' and r['new_label'] in ['SUPPORT','WEAK_SUPPORT']]
    text.extend(['### Previous false rejection examples (training diagnosis)'])
    for r in changed[:5]:text.extend([r['claim'],r['quote'],f"Old: {r['old_label']}; new: {r['new_label']}; support={r['support']:.3f}, contradiction={r['contradiction']:.3f}, coverage={r['coverage']:.3f}, cosine={r['cosine']:.3f}. See stored entity/context diagnostics; no dev-based retuning."])
    text.extend(['## 17. Component comparison','Vanilla / NLI-only / full auditing, all actually executed. NLI-only includes the existing claim-based retrieval and quote-recovery path, but neither counter-search nor wording probes. Therefore this isolates that verification layer, not NLI inference alone.',
      '| System | Coverage | NLI support | NLI groundedness | Automated stance agreement | Mean seconds |','|---|---:|---:|---:|---:|---:|'])
    for name,s in summaries.items():text.append('| '+' | '.join([name,fmt(s['Coverage']),fmt(s['Claim Support Rate (NLI)']),fmt(s['Groundedness (NLI)']),fmt(s['Automated answer-stance accuracy']),fmt(s['Average Latency (s)'])])+' |')
    text.extend(['Actual full-system decision distribution: '+json.dumps(summaries['ragstress']['decisions']),
      '### Incremental contribution of the stress tests',
      'In this executed subset, NLI-only and full auditing have identical aggregate claim support, groundedness, answer coverage, fully-supported answer incidence and overall automated stance agreement. Full auditing is slower. Therefore the results do not establish an incremental answer-quality benefit from counter-search or wording probes beyond the existing verification/recovery layer. The probes still provide inspectable evidence diagnostics. No query crosses the severe mean-overlap threshold; a few individual pairs do.',
      '### Directional failures in the stance diagnostic',
      f"The automated confusion matrices show gold-SUPPORT agreement {agreed('vanilla',0)} versus {agreed('ragstress',0)}, and gold-CONTRADICT agreement {agreed('vanilla',1)} versus {agreed('ragstress',1)}. These class-specific outcomes must not be hidden by the overall diagnostic. The same NLI judge infers answer relation, so these are model-based failure signals requiring human review.",
      '## 18. Limitations','Small balanced exploratory subset does not represent natural label prevalence; prior dev exposure; generic NLI scientific-domain errors; same-model auditing/scoring circularity; no independent generated-claim gold; heuristic sentence extraction and alias rules; incomplete literature/corpus annotations; wording-probe equivalence not human-reviewed; extra retrieval budget for counter-search; warm CPU timing and quote recovery confounds. Verbatim quotation recovery naturally improves text-to-evidence entailment without establishing independent scientific truth. Human review is still required.',
      '## 19. Measured conclusion',
      f"Initial IR is identical, as required. Model-based groundedness changes from {summaries['vanilla']['Groundedness (NLI)']:.3f} to {summaries['ragstress']['Groundedness (NLI)']:.3f}; answer coverage from {summaries['vanilla']['Coverage']:.3f} to {summaries['ragstress']['Coverage']:.3f}. Mean latency changes from {summaries['vanilla']['Average Latency (s)']:.3f}s to {summaries['ragstress']['Average Latency (s)']:.3f}s ({summaries['ragstress']['RAGStress Overhead (%)']:.1f}% overhead). Exact McNemar p-values: fully supported answer incidence {support_p:.9f}; automated stance agreement {stance_p:.9f}. Interpret these with their stated diagnostic limitations. NLI-only matches the main aggregate quality metrics at lower cost; stress-test-specific gains are not established here. These results describe this corpus and automated assessment, not universal trustworthiness or verified free-form answer correctness.",
      '## Reproduction','Run `python run_evaluation.py`. Verified matching checkpoints resume the fixed experiment; use `--no-resume` for an entirely new timed rerun. Freeze checks prevent silently changing the runtime during this study. Raw results and eight figures are included.',
      '## Implementation handoff','Calibration details, old/current gates and reasons: docs/CALIBRATION.md. Created: src/vanilla.py, evaluation/study.py, evaluation/analysis.py, evaluation/evaluation_dataset.json, run_evaluation.py, calibration artifacts, tests/test_calibrated_policy.py and tests/test_study.py. Modified: src/config.py, src/stance.py, src/trust.py, src/pipeline.py, src/evaluation.py, src/cli.py, app.py, src/presentation.py, important IR/presentation/answer-repair tests, README and design/experiment/UI/AI-use/video documentation. The obsolete dependency stage and its historical active outputs were deleted. An archive is outside the active submission. Backend sanity runs and 34 tests passed; UI rendering checks use recorded real outputs and are not presented as a live performance experiment.',
      '## Computational cost','All compared runs use local CPU models; no remote API is called. Latency measures computational overhead, not electricity or dollar cost. Model loading/indexing and post-hoc evaluation are outside timed answering. Stage-wise timing appears in the comparison table.'])
    (OUT/'final_report.md').write_text('\n\n'.join(text),encoding='utf8')
