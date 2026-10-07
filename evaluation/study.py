import hashlib,json,random,time
from pathlib import Path
import numpy as np
from src.config import ROOT,DATA,SEED,NLI_THRESHOLD,CONTRADICTION_THRESHOLD
from src.data import jsonl
from src.claims import extract_claims
from src.evaluation import metrics
from src.stability import jaccard
from src.vanilla import VanillaRAG
from src.pipeline import Pipeline

STUDY=ROOT/'evaluation'
OUT=STUDY/'results'
NA='Not available — insufficient gold annotation'

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')

def assert_frozen():
    manifest=json.loads((ROOT/'calibration/frozen_manifest.json').read_text(encoding='utf8'))
    for name,expected in manifest['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Frozen runtime changed: {name}. Recalibrate and freeze explicitly before a new study.')
    return manifest

def dataset(limit=50):
    path=STUDY/'evaluation_dataset.json'
    if path.exists():
        rows=json.loads(path.read_text(encoding='utf8'))
        if len(rows)!=limit:
            raise ValueError('Fixed dataset size differs; preserve this study and create a separately named experiment.')
        return rows
    corpus={str(r['doc_id']):r for r in jsonl(DATA/'original/corpus.jsonl')}
    candidates={'SUPPORT':[],'CONTRADICT':[]}
    for r in jsonl(DATA/'original/claims_dev.jsonl'):
        labels={a['label'] for annotations in r.get('evidence',{}).values() for a in annotations}
        if len(labels)!=1:continue
        label=next(iter(labels));evidence=[]
        for sid,annotations in r['evidence'].items():
            for a in annotations:
                for index in a['sentences']:
                    evidence.append({'doc_id':sid,'sentence_index':index,'text':corpus[sid]['abstract'][index],'label':a['label']})
        candidates[label].append({'id':str(r['id']),'query':r['claim'],'gold_label':label,
            'gold_doc_ids':list(r['evidence']),'gold_evidence':evidence,'split':'original-dev'})
    rng=random.Random(SEED)
    rows=[]
    for label in ['SUPPORT','CONTRADICT']:
        pool=sorted(candidates[label],key=lambda r:int(r['id']));rng.shuffle(pool)
        rows.extend(pool[:limit//2+(limit%2 if label=='SUPPORT' else 0)])
    if len(rows)!=limit:raise ValueError('Not enough labelled dev examples')
    rng.shuffle(rows);save(path,rows)
    return rows

def final_claims(result):
    claims=[]
    for c in result['claims']:
        if c['decision']=='ABSTAIN':continue
        if c['decision']=='REWRITE':
            rows=[r for r in c['evidence'] if r['label']=='CONTRADICT']
            if rows:claims.append(max(rows,key=lambda r:r['contradiction'])['quote'])
        else:claims.append(c['claim'])
    return list(dict.fromkeys(claims))

def judge_claims(claims,hits,classifier):
    rows=[]
    for claim in claims:
        evidence=classifier.classify(claim,hits)
        support=any(e['label'] in ('SUPPORT','WEAK_SUPPORT') for e in evidence)
        contra=any(e['label']=='CONTRADICT' for e in evidence)
        rows.append({'claim':claim,'supported':support,'contradicted':contra,
                     'unsupported':not support and not contra,'evidence':evidence})
    return rows

def answer_stance(query,claims,classifier,abstained=False,contested=False):
    # This scores output-to-original-claim alignment, NOT factual correctness of novel claims.
    if abstained or not claims:return {'predicted':'ABSTAIN','scores':None}
    if contested:return {'predicted':'UNRESOLVED','scores':None}
    scores=classifier.predict([(' '.join(claims),query)])[0]
    label='SUPPORT' if scores[1]>=NLI_THRESHOLD else 'CONTRADICT' if scores[0]>=CONTRADICTION_THRESHOLD else 'UNRESOLVED'
    return {'predicted':label,'scores':list(map(float,scores))}

def normalize(text):return ' '.join(text.casefold().split())

def gold_recall(hits,example):
    gold={(e['doc_id'],e['sentence_index']):e['text'] for e in example['gold_evidence']}
    found=sum(any(h['source_id']==sid and normalize(text) in normalize(h['passage']) for h in hits)
              for (sid,_),text in gold.items())
    return found/len(gold) if gold else None

def reset(retriever,classifier):
    retriever.dense.query_cache.clear()
    classifier.cache.clear();classifier.embedding_cache.clear()

def run_study(retriever,generator,classifier,limit=50,resume=True):
    manifest=assert_frozen();examples=dataset(limit)
    signature=hashlib.sha256(json.dumps({'manifest':manifest,'dataset':examples},sort_keys=True).encode()).hexdigest()
    protocol={'signature':signature,'examples':len(examples),'seed':SEED,'split':'original-dev',
        'gold_used_at_runtime':False,'system_order':'rotating vanilla / NLI-only / full',
        'latency':'warm model process, clear query/NLI/semantic caches before each timed system; model loading excluded',
        'groundedness':'same calibrated NLI judge, same initial Top-5 pool for both final outputs; model-based diagnostic',
        'correctness':'Novel generated sentences lack gold labels. Primary correctness is unavailable; automated answer-stance alignment is a separate diagnostic.',
        'prior_exposure':'Exploratory: earlier prototype development inspected some dev examples; not a pristine unseen test.',
        'freeze':manifest}
    # Never relabel historical outputs before confirming checkpoint compatibility.
    if resume:
        for index,example in enumerate(examples):
            checkpoint=OUT/'raw'/f"{index:03d}-{example['id']}.json"
            if checkpoint.exists() and json.loads(checkpoint.read_text(encoding='utf8'))['signature']!=signature:
                raise RuntimeError('Checkpoint describes the earlier answering version. Use --no-resume for the paragraph revision; the prior evaluated snapshot is preserved separately.')
    save(OUT/'protocol.json',protocol)
    systems={'vanilla':VanillaRAG(retriever,generator),
             'nli_only':Pipeline(retriever,generator,classifier,False,False),
             'ragstress':Pipeline(retriever,generator,classifier)}
    # Actual untimed warmup; never part of the evaluation dataset.
    generator.generate('What potential does antiretroviral therapy have to prevent HIV-associated tuberculosis?',
                       retriever.retrieve('antiretroviral therapy tuberculosis',5))
    records=[]
    for index,example in enumerate(examples):
        assert_frozen()
        checkpoint=OUT/'raw'/f"{index:03d}-{example['id']}.json"
        if resume and checkpoint.exists():
            record=json.loads(checkpoint.read_text(encoding='utf8'))
            if record['signature']!=signature:raise RuntimeError('Checkpoint describes the earlier answering version. Use --no-resume for the paragraph revision; the prior evaluated snapshot is preserved separately.')
            records.append(record);print(f"Resumed {index+1}/{len(examples)}",flush=True);continue
        record={'signature':signature,'example':example,'systems':{}}
        names=list(systems);order=names[index%3:]+names[:index%3]
        for name in order:
            reset(retriever,classifier)
            record['systems'][name]=systems[name].run(example['query'],'hybrid',5)
        vanilla=record['systems']['vanilla'];full=record['systems']['ragstress']
        for name in ['nli_only','ragstress']:
            other=record['systems'][name]
            assert other['retrieved']==vanilla['retrieved'],'Initial retrieval differs'
            assert other['initial_answer']==vanilla['generated_answer'],'Deterministic initial drafts differ'
        additional= retriever.retrieve(example['query'],10,'hybrid')
        for name,result in record['systems'].items():
            content=extract_claims(result['generated_answer'],100)[0] if name=='vanilla' else final_claims(result)
            initial=result['retrieved'];ir=metrics([h['source_id'] for h in initial],{s:1 for s in example['gold_doc_ids']},5)
            ir.update({'Recall@1':metrics([h['source_id'] for h in initial],{s:1 for s in example['gold_doc_ids']},1)['Recall@1']})
            ir.update(metrics([h['source_id'] for h in additional],{s:1 for s in example['gold_doc_ids']},10))
            judged=judge_claims(content,initial,classifier)
            abstained=name!='vanilla' and result['decision']=='ABSTAIN'
            contested=name!='vanilla' and any(c['status']=='CONTESTED' for c in result['claims'])
            stance=answer_stance(example['query'],content,classifier,abstained,contested)
            result['evaluation']={'content_claims':content,'claim_judgments':judged,'ir':ir,
                'gold_sentence_recall':gold_recall(initial,example),'answer_stance_diagnostic':stance,
                'gold_stance_agreement':stance['predicted']==example['gold_label'],
                'answered':bool(content) and not abstained,
                'fully_supported_answer':bool(judged) and all(r['supported'] and not r['contradicted'] for r in judged),
                'evidence_precision':NA,'freeform_answer_correctness':NA}
        # Gold-refuting-source diagnostic assesses the original claim, not a changed generated claim.
        counter_hits={}
        for c in full['claims']:
            for run in c['counter_evidence']['runs']:
                for hit in run['hits']:counter_hits[(hit['source_id'],hit['chunk_id'])]=hit
        refute=example['gold_label']=='CONTRADICT'
        gold_ids=set(example['gold_doc_ids'])
        def detected(hits):
            evidence=classifier.classify(example['query'],hits)
            return any(r['label']=='CONTRADICT' and r['source_id'] in gold_ids for r in evidence)
        initial_detected=detected(vanilla['retrieved']) if refute else None
        counter_detected=detected(list(counter_hits.values())) if refute else None
        record['gold_refutation_diagnostic']={'eligible':refute,'vanilla':initial_detected,
            'counter_only':counter_detected,
            'ragstress':(initial_detected or counter_detected) if refute else None,
            'expanded_gold_sentence_recall':gold_recall(vanilla['retrieved']+list(counter_hits.values()),example)}
        sets=[full['stability']['base_ids']]+[v['source_ids'] for v in full['stability']['variants']]
        record['stability_pairwise']=[jaccard(a,b) for i,a in enumerate(sets) for b in sets[i+1:]]
        record['counter_discovery_claims']=sum(any(e['label']=='CONTRADICT' for e in c['counter_evidence']['evidence']) for c in full['claims'])
        record['counter_tested_claims']=len(full['claims'])
        base_claims=vanilla['evaluation']['content_claims']
        record['unqualified_draft']=bool(base_claims) and not any(any(w in c.casefold() for w in ['may ','might ','could ','suggest','insufficient','not enough']) for c in base_claims)
        record['potential_false_confidence']=record['unqualified_draft'] and record['counter_discovery_claims']>0
        save(checkpoint,record);records.append(record)
        print(f"Completed {index+1}/{len(examples)} · {example['id']} · {full['decision']} · {full['elapsed_seconds']:.1f}s",flush=True)
    assert_frozen();save(OUT/'raw_results.json',records)
    return records,protocol
