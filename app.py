"""Streamlit presentation; the pipeline owns retrieval and evidence decisions."""
import json
import os
import time
import streamlit as st
from src.cli import build
from src.config import (DATA, RESULTS, MODEL_REVISIONS, NLI_THRESHOLD, STABILITY_THRESHOLD,
    WEAK_SUPPORT_THRESHOLD, CONTRADICTION_THRESHOLD, TOPICAL_COVERAGE_THRESHOLD, SEMANTIC_ALIGNMENT_THRESHOLD)
from src.generator import Generator
from src.pipeline import Pipeline
from src.stance import Stance
from src.presentation import (DECISIONS, source_catalog, source_label, citations, highlighted_quote,
    settings_snapshot, result_is_stale, evidence_summary, ranking_rows, decision_reason, claim_reason, safe_export, markdown_audit,
    answer_paragraph, approved_source_documents)

st.set_page_config(page_title='RAGStress · Check the evidence', page_icon='🔎', layout='wide')
METHODS = {'hybrid': 'Hybrid · keywords + meaning', 'bm25': 'BM25 · keyword matching',
           'tfidf': 'TF-IDF · weighted keyword similarity', 'dense': 'Dense · semantic similarity'}
GENERATORS = {'local': 'Local model · no API key', 'extractive': 'Retrieved quotations', 'remote': 'External API'}

@st.cache_resource(show_spinner=False)
def resources(dense_enabled, generator_mode, identity):
    retriever = build(dense_enabled)
    warning = None
    try:
        classifier = Stance(retriever)
    except Exception as problem:
        classifier = None
        warning = f'The evidence-checking model could not load ({type(problem).__name__}). Sources can still be retrieved, but claims cannot be verified. Check model setup in README and retry.'
    return Pipeline(retriever, Generator(generator_mode), classifier), warning

@st.cache_data(show_spinner=False)
def corpus_count(path, modified):
    with open(path, encoding='utf8') as stream:
        return sum(bool(line.strip()) for line in stream)

def change_mode():
    st.session_state['query'] = ''

def select_source(sid):
    st.session_state['selected_source'] = str(sid)

def quote(text, query):
    st.markdown('<p>' + highlighted_quote(text, query) + '</p>', unsafe_allow_html=True)

def evidence_cards(rows, label, catalog, query, key):
    names = {'WEAK_SUPPORT': 'Moderate supporting evidence', 'SUPPORT': 'Supporting evidence', 'CONTRADICT': 'Conflicting evidence', 'NEUTRAL': 'Inconclusive evidence'}
    st.markdown(f"**{names[label]} · {len(rows)} passage assessments**")
    if not rows:
        st.caption('None found in the evaluated passages.')
        return
    if label == 'NEUTRAL':
        st.caption('Retrieved, but does not clearly support or contradict this claim after the audit gates.')
    if len(rows) > 4:
        page = st.selectbox('Evidence page (4 cards per page)', range(1, (len(rows) + 3) // 4 + 1), key=key)
        rows = rows[(page - 1) * 4:page * 4]
    for index, row in enumerate(rows):
        sid = str(row['source_id'])
        with st.container(border=True):
            st.markdown(f"**{source_label(sid, catalog)} — {catalog[sid]['title']}**")
            st.caption(f"Model assessment: {label} · passage {row.get('chunk_id', 'unavailable')}")
            quote(row['quote'], query)
            st.button(f"Read full Source {catalog[sid]['number']}", key=f'{key}-source-{index}',
                      on_click=select_source, args=(sid,), help='Selects this source in the Retrieval details tab.')
            with st.expander('Model scores and audit gates'):
                st.write({k: v for k, v in row.items() if k not in ('quote', 'source_id', 'chunk_id')})
                st.caption('Model scores are not calibrated probabilities that the scientific claim is true. Labels can be wrong.')

def show_evaluation():
    comparison_path = RESULTS.parent / 'evaluation/results/comparison.json'
    if comparison_path.exists():
        st.subheader('Vanilla RAG versus calibrated RAGStress')
        st.caption('Historical 50-case comparison · same initial hybrid retrieval and local generator · Top-K 5. The tested backend was frozen before that experiment.')
        protocol_file = comparison_path.with_name('protocol.json')
        if protocol_file.exists():
            import hashlib
            recorded = json.loads(protocol_file.read_text(encoding='utf8')).get('freeze', {}).get('files', {})
            if any(not (RESULTS.parent / name).exists() or hashlib.sha256((RESULTS.parent / name).read_bytes()).hexdigest() != digest for name, digest in recorded.items()):
                st.warning('These results describe the earlier answer format. The current paragraph-generation revision has not yet been benchmarked; do not attribute these scores to it.')
        measured = json.loads(comparison_path.read_text(encoding='utf8'))
        display_rows = [{key: f'{value:.4f}' if isinstance(value, (int, float)) else str(value) if value is not None else '—'
                         for key, value in row.items()} for row in measured]
        st.dataframe(display_rows, hide_index=True, width='stretch')
        st.info('Claim support and groundedness are automated NLI diagnostics. Answer-stance alignment is a separate diagnostic; free-form correctness lacks gold annotation. Coverage and latency show the tradeoff.')
        report_path = comparison_path.with_name('final_report.md')
        if report_path.exists():
            st.download_button('Download full measured evaluation', report_path.read_text(encoding='utf8'), 'ragstress-evaluation.md', 'text/markdown')
        st.divider()
    st.subheader('Dataset retrieval benchmark')
    st.caption('These measurements describe retrieval across the dataset, not confidence in this answer.')
    path = RESULTS / 'retrieval-test.json'
    if not path.exists():
        st.info('Benchmark results are not available yet. Run the evaluation described in README to create measured results.')
        return
    try:
        evaluation = json.loads(path.read_text(encoding='utf8'))
        records = evaluation['summary']
        st.write(f"SciFact · {evaluation['corpus_sources']:,} abstracts · query counts: " +
                 ', '.join(f"{r['method']}: {r['queries']}" for r in records) + f" · test split · K={evaluation['k']}")
        st.dataframe([{k: v for k, v in r.items() if k not in ('seconds', 'queries')} for r in records], hide_index=True, width='stretch')
        st.markdown('**How to read the metrics**')
        st.write('Precision: how many top results are relevant. Recall: how much annotated relevant evidence was recovered. MRR: how early the first relevant result appears. nDCG: how well relevant results are ranked. Higher is better for these metrics.')
        if evaluation.get('hybrid_minus_bm25'):
            st.markdown('**Hybrid minus BM25 · measured differences**')
            st.dataframe([{'Metric': row['metric'], 'Difference': row['mean_delta'],
                '95% interval lower': row['ci95'][0], '95% interval upper': row['ci95'][1]}
                for row in evaluation['hybrid_minus_bm25']], hide_index=True, width='stretch')
            st.caption('Paired-bootstrap intervals. An interval including zero does not establish an improvement.')
        st.caption(evaluation['protocol'])
        st.info('Human evaluation of generated-answer correctness is pending. Retrieval metrics do not measure end-to-end truthfulness.')
        with st.expander('Benchmark graph and reproducibility notes'):
            image = RESULTS / 'retrieval-test.png'
            if image.exists():
                st.image(str(image))
            st.write('Runtime measurements reuse cached query embeddings; they are not a fair comparison of latency. See docs/EXPERIMENTS.md for methods and limitations.')
    except (ValueError, KeyError, OSError):
        st.warning('The benchmark file could not be read. Re-run evaluation; no scores have been substituted.')

st.title('RAGStress')
st.write('Ask a scientific question. Inspect its sources and see whether the answer survives evidence stress tests.')
corpus_path = DATA / 'scifact/corpus.jsonl'
if corpus_path.exists():
    st.caption(f"Searches {corpus_count(str(corpus_path), corpus_path.stat().st_mtime_ns):,} SciFact scientific abstracts. Answers are limited to evidence in this corpus.")
else:
    st.warning('The SciFact dataset is not installed. Follow the dataset download step in README before running a check.')
mode = st.radio('What would you like to do?', ['Ask a question', 'Check a claim'], horizontal=True,
                key='input_mode', on_change=change_mode)
claim_mode = mode == 'Check a claim'
st.caption('Your supplied claim → Evidence audit → Final assessment' if claim_mode else
           'Your question → Draft from retrieved passages → Extracted claims → Final answer')
st.write('We audit exactly the claim you supply. Conflicting evidence may change the final assessment; your original claim stays visible.' if claim_mode else
         'The selected generator drafts an answer using retrieved passages. The system then checks its factual claims against corpus evidence.')
with st.sidebar:
    st.header('About this check')
    st.write('Sources first. Every decision has an evidence trail.')
    with st.expander('Advanced settings'):
        method = st.selectbox('Retrieval method', list(METHODS), format_func=METHODS.get, key='method')
        st.caption('Hybrid combines keyword and semantic ranks. Sparse-only methods use a weaker vocabulary-based query relevance gate.')
        remote_ready = bool(os.environ.get('RAG_LLM_URL') and os.environ.get('RAG_LLM_MODEL'))
        options = ['local', 'extractive'] + (['remote'] if remote_ready else [])
        if st.session_state.get('generator') not in options:
            st.session_state['generator'] = 'local'
        generator_mode = st.selectbox('Answer generator', options, format_func=GENERATORS.get,
                                     key='generator', disabled=claim_mode)
        st.caption('Local model: runs on this computer with no API key. Retrieved quotations: no LLM generation.')
        if remote_ready:
            st.caption('External API is configured, but connectivity is not verified. Selecting it sends your question and selected passages to that endpoint.')
        else:
            st.caption('External API unavailable: configure RAG_LLM_URL and RAG_LLM_MODEL, and RAG_LLM_KEY if the endpoint needs authentication. Secrets are never displayed.')
        k = st.slider('Number of ranked sources (Top-K)', 3, 10, 5, key='top_k')
        rewrites_text = st.text_area('Supply alternative wording (optional)', key='rewrites', help='One query per line. Automatic probes run when this is blank.')
        st.caption('Automatic probes are not guaranteed semantic equivalents. Supplied variants should be reviewed for equivalence.')
        st.markdown('**Fixed prototype thresholds**')
        st.caption(f'Strong support: {NLI_THRESHOLD} · moderate support: {WEAK_SUPPORT_THRESHOLD} · contradiction: {CONTRADICTION_THRESHOLD} · severe wording overlap: {STABILITY_THRESHOLD}. Heuristic gates, not truth probabilities.')
        st.caption(f'Claim alignment: weighted term coverage ≥ {TOPICAL_COVERAGE_THRESHOLD} or semantic cosine ≥ {SEMANTIC_ALIGNMENT_THRESHOLD}. Entity and passage-context checks still apply.')
    with st.expander('Models and limitations'):
        st.write('Default local generator: FLAN-T5-small. Dense retrieval: MiniLM. Stance: DeBERTa NLI. Model versions are included in the audit.')
        st.write('Neural stance labels can fail on scientific language. Exact-duplicate groups do not establish study independence. An attributed quote is not independently verified truth.')
if 'query' not in st.session_state:
    st.session_state['query'] = ''
query = st.text_input('Claim to check' if claim_mode else 'Scientific question', key='query',
                      placeholder='Enter one factual claim' if claim_mode else 'Ask about evidence in the scientific corpus')
submitted = st.button('Check claim' if claim_mode else 'Check answer', type='primary', key='submit_check')
st.caption('Use Check answer / Check claim to submit. Keyboard: Tab to the button, then Enter. Editing the input marks any completed result as previous when the edit is applied.')
rewrites = [line.strip() for line in rewrites_text.splitlines() if line.strip()] or None
identity = '|'.join([os.environ.get('RAG_LLM_URL', ''), os.environ.get('RAG_LLM_MODEL', ''),
                     os.environ.get('RAG_LLM_KEY', '')]) if generator_mode == 'remote' else ''
thresholds = {'strong_support': NLI_THRESHOLD, 'moderate_support': WEAK_SUPPORT_THRESHOLD,
              'contradiction': CONTRADICTION_THRESHOLD, 'query_cosine': .30,
              'wording_overlap': STABILITY_THRESHOLD, 'claim_term_coverage': TOPICAL_COVERAGE_THRESHOLD,
              'semantic_alignment': SEMANTIC_ALIGNMENT_THRESHOLD}
current = settings_snapshot(query, claim_mode, method, generator_mode, k, rewrites, thresholds, identity)
current['model_revisions'] = MODEL_REVISIONS
if submitted:
    st.session_state.pop('last_failure', None)
    if not query.strip():
        st.session_state['last_failure'] = {'query': query, 'message': 'Enter a question or claim before running a check.', 'type': 'EmptyInput'}
    else:
        started = time.perf_counter()
        try:
            with st.status('Loading resources', expanded=True) as status:
                def progress(stage):
                    elapsed = time.perf_counter() - started
                    status.update(label=f'{stage} · {elapsed:.1f}s elapsed')
                    status.write(f'{stage} · {elapsed:.1f}s elapsed')
                status.write('First use loads indexes and models. Later checks reuse cached resources; the generation model may load during draft generation.')
                pipeline, warning = resources(method in ('hybrid', 'dense'), generator_mode,
                                              (current['remote_config_fingerprint'], tuple(MODEL_REVISIONS.items())))
                result = pipeline.run(query.strip(), method, k, claim_mode, rewrites, progress=progress)
                result['run_settings'] = current
                result['verification_warning'] = warning
                st.session_state['result'] = result
                st.session_state['result_settings'] = current
                st.session_state.pop('selected_source', None)
                status.update(label=f'Check complete · {time.perf_counter() - started:.1f}s', state='complete', expanded=False)
        except Exception as problem:
            try:
                status.update(label='This check could not complete', state='error', expanded=False)
            except NameError:
                pass
            message = ('The dataset could not be found. Follow the download step in README and retry.' if isinstance(problem, FileNotFoundError)
                       else 'This check could not complete. Check dataset/model setup in README and retry. Any previous completed run is retained below.')
            st.session_state['last_failure'] = {'query': query, 'message': message, 'type': type(problem).__name__}
failure = st.session_state.get('last_failure')
if failure:
    st.error(f"Last attempt failed for input: {failure['query'] or '(empty)'}. {failure['message']}")
    with st.expander('Failure diagnostics'):
        st.write('Exception type: ' + failure['type'])
        st.caption('Sensitive exception messages and endpoint details are omitted.')
result = st.session_state.get('result')
if result:
    catalog = source_catalog(result)
    previous = bool(failure) or result_is_stale(st.session_state.get('result_settings'), current)
    if previous:
        st.warning('Previous completed result — it does not represent the edited input/settings or a failed new attempt. Submit a new check to update it.')
    st.divider()
    st.caption('PREVIOUS COMPLETED RUN' if previous else 'COMPLETED RUN · SUBMITTED INPUT')
    st.markdown(f"**{result['query']}**")
    st.caption(f"{'Check a claim' if result.get('claim_mode') else 'Ask a question'} · {METHODS.get(result['method'], result['method'])} · {len(result['retrieved'])} initially retrieved sources · {result['elapsed_seconds']:.1f}s pipeline time")
    symbol, title = DECISIONS[result['decision']]
    st.subheader(f'{symbol} {title}')
    st.caption('Decision code: ' + result['decision'])
    st.markdown(answer_paragraph(result))
    st.subheader('Approved source documents')
    approved_documents = approved_source_documents(result)
    st.caption('Documents used for the final answer after the evidence audit. Conflicting and inconclusive passages remain in Claims & evidence.')
    if not approved_documents:
        st.info('No source documents were approved for an answer.')
    for document in approved_documents:
        with st.container(border=True):
            st.markdown('**' + document['title'] + '**')
            st.caption(f"Source {document['number']} · ID {document['source_id']} · {', '.join(document['roles'])} · Claims {', '.join(map(str, document['claim_numbers']))}")
            with st.expander('Approved evidence excerpts'):
                for excerpt in document['quotes']:
                    quote(excerpt, result['query'])
            st.button('Read this document', key=f"approved-source-{document['source_id']}",
                      on_click=select_source, args=(document['source_id'],))
    st.markdown('**Why this decision?**')
    st.write(decision_reason(result))
    if result.get('verification_warning'):
        st.warning(result['verification_warning'])
    if result.get('warning') and not result['warning'].startswith('Initial references'):
        if 'Generator unavailable' in result['warning']:
            st.warning('The selected generator could not run. A retrieved quotation was used instead. Check the local model setup or external endpoint configuration to restore generation.')
        else:
            st.caption(result['warning'])
    if result['decision'] == 'REWRITE':
        with st.container(border=True):
            st.markdown('**Before → After**')
            st.caption('Original supplied claim' if result.get('claim_mode') else 'Original draft — provisional context references, not verified citations')
            st.write(citations(result['initial_answer'], catalog))
            st.caption('Replacement / final assessment')
            st.write(answer_paragraph(result))
            for rejected in result.get('superseded_claims', []):
                st.caption(f"Rejected wording: {rejected['claim']} — {rejected['reason']}")
            if result.get('answer_repair'):
                st.info('The draft was corrected using an attributed quotation. The replacement has its own evidence assessment below; correction does not imply robust or independent support.')
            else:
                st.info('The claim was not retained because retrieved evidence contradicted it. The replacement is an attributed passage, not a separately certified conclusion.')
    summary = evidence_summary(result)
    columns = st.columns(3)
    columns[0].metric('Supporting documents', 'Unavailable' if summary['unverified'] else summary['support_documents'])
    columns[0].caption('Verification unavailable; no support labels assigned.' if summary['unverified'] else f"{summary['support_passages']} unique passages · {summary['support_groups']} duplicate-aware groups")
    columns[1].metric('Conflicting documents', 'Unavailable' if summary['unverified'] else summary['counter_documents'])
    columns[1].caption('Verification unavailable; no contradiction labels assigned.' if summary['unverified'] else f"{summary['counter_passages']} unique passages")
    stability = result['stability']
    columns[2].metric('Wording overlap', f"{stability['mean_jaccard']:.2f}" if stability['variants'] else 'Not tested')
    columns[2].caption('Mean Jaccard of source sets, not a truth score.')
    st.caption('Evidence counts are unions across audited claims. Distinct documents and duplicate groups are not proof of independent studies.')
    if len(result['claims']) > 1:
        st.write('Claim outcomes: ' + ' · '.join(f"{i}: {c['decision']}" for i, c in enumerate(result['claims'], 1)))
claims_tab, stress_tab, retrieval_tab, evaluation_tab = st.tabs(['Claims & evidence', 'Stress tests', 'Retrieval details', 'Evaluation'])
with evaluation_tab:
    show_evaluation()
if not result:
    with claims_tab:
        st.info('Submit a question or claim to see an answer and its evidence.')
    with stress_tab:
        st.write('The system searches for conflicting evidence and probes alternative wording.')
    with retrieval_tab:
        st.write('Ranked documents, retrieval scores, passages and dictionary statistics appear after a check.')
else:
    with claims_tab:
        st.caption('Your claim → evidence audit' if result.get('claim_mode') else
                   'Draft → sentence/semicolon candidates → audited claims. Candidates can contain multiple assertions.')
        st.caption('Actual generator: ' + result['generation_mode'])
        with st.expander('Original draft and model provenance'):
            st.write(citations(result['initial_answer'], catalog))
            st.caption('Draft references identify supplied context; they are not verified support. Unmapped references are labelled explicitly.')
            st.write('Stance model: ' + (result.get('stance_model') or 'Unavailable'))
            st.json(safe_export(result.get('run_settings', {})), expanded=False)
        if result.get('unexamined_sentences_omitted'):
            st.warning('Additional generated sentences were omitted because they were not audited.')
        if result['claims']:
            ci = st.selectbox('Claim to inspect', range(len(result['claims'])),
                               format_func=lambda i: f"Claim {i+1} · {result['claims'][i]['decision']}", key='claim_inspect')
            audit = result['claims'][ci]
            st.markdown(f"**Claim {ci + 1}**")
            st.write(audit['claim'])
            st.caption('Provenance: retrieved supporting detail, separately audited' if audit.get('origin') == 'retrieved_context' else
                       'Provenance: attributed replacement quote' if result.get('answer_repair') else
                       'Provenance: supplied by you' if result.get('claim_mode') else 'Provenance: extracted from the generated draft')
            st.write(f"{DECISIONS[audit['decision']][1]} · {audit['status']}")
            st.write(claim_reason(audit))
            for label in ['SUPPORT', 'WEAK_SUPPORT', 'CONTRADICT', 'NEUTRAL']:
                rows = [r for r in audit['evidence'] if r['label'] == label]
                rows.sort(key=lambda r: -r.get('support' if label in ('SUPPORT', 'WEAK_SUPPORT') else 'contradiction' if label == 'CONTRADICT' else 'neutral', 0))
                evidence_cards(rows, label, catalog, audit['claim'], f'evidence-{ci}-{label}')
            with st.expander('Claim component scores'):
                st.json(audit['components'])
        else:
            st.info('No factual claim could be extracted for auditing.')
    with stress_tab:
        st.subheader('Searches for conflicting evidence')
        st.caption('All results come from the corpus. Query wording alone does not make a passage contradictory.')
        for ci, audit in enumerate(result['claims']):
            with st.expander(f'Actual counter-searches · Claim {ci+1}'):
                for run in audit['counter_evidence']['runs']:
                    st.write(run['query'])
                    st.dataframe(ranking_rows(run['hits'], catalog), hide_index=True, width='stretch')
                st.caption('Inspect classified quotations in Claims & evidence and full passages in Retrieval details.')
        st.subheader('Does retrieval change with wording?')
        st.caption(stability['kind'])
        st.write('Original: ' + result['query'])
        st.caption('Original ranked IDs: ' + ', '.join(stability['base_ids']))
        for v in stability['variants']:
            with st.container(border=True):
                st.write(v['query'])
                st.caption(f"Jaccard overlap: {v['jaccard']:.3f} · ranked IDs: {', '.join(v['source_ids'])}")
        st.write('Higher overlap means similar retrieved source sets. Low overlap means sensitivity to wording, not automatically an incorrect answer. Automatic probes need equivalence review.')
    with retrieval_tab:
        st.subheader('Initially ranked sources')
        st.dataframe(ranking_rows(result['retrieved'], catalog), hide_index=True, width='stretch')
        st.caption('BM25 is an unbounded keyword-ranking score; dense and TF-IDF use cosine similarity. RRF combines ranks, not raw scores. A dash means unavailable.')
        if catalog:
            if st.session_state.get('selected_source') not in catalog:
                st.session_state['selected_source'] = next(iter(catalog))
            sid = st.selectbox('Read a source from any audit stage', list(catalog), key='selected_source',
                               format_func=lambda s: source_label(s, catalog) + ' — ' + catalog[s]['title'])
            source = catalog[sid]
            st.markdown('**' + source['title'] + '**')
            st.caption(source_label(sid, catalog))
            st.caption('Source cards select this reader; open Retrieval details to read them. This avoids unreliable jump links across tabs.')
            for passage in source['passages']:
                quote(passage, result['query'])
            if not source['passages']:
                for quotation in source['quotes']:
                    quote(quotation, result['query'])
                if not source['quotes']:
                    st.caption('This source was recorded by ID only; no passage was returned for display.')
        with st.expander('IR dictionary: tokens, document frequency, IDF and postings'):
            st.caption('Unicode-normalized, case-folded text. BM25 uses an explicit inverted index; DF counts documents containing a term, and IDF increases with rarity.')
            st.json(result['query_terms'])
        with st.expander('Complete technical audit'):
            st.json(safe_export(result), expanded=False)
    st.divider()
    st.markdown('**Save this exact completed run**')
    st.download_button('Readable audit · Markdown', markdown_audit(result), 'ragstress-audit.md', 'text/markdown')
    st.download_button('Technical audit · JSON', json.dumps(safe_export(result), indent=2), 'ragstress-audit.json', 'application/json')
