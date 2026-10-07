import json
import streamlit as st
from src.cli import build
from src.config import RESULTS, NLI_THRESHOLD, STABILITY_THRESHOLD
from src.generator import Generator
from src.pipeline import Pipeline
from src.stance import Stance

st.set_page_config(page_title='RAGStress', layout='wide')
st.title('RAGStress | Counterfactual evidence audits')
st.caption('T1 prototype · retrieval scores are visible · audit decisions are heuristic, not calibrated truth probabilities')
st.caption('Backend v2 · unsupported generated wording can be replaced with audited corpus evidence')

@st.cache_resource
def resources(dense_enabled, generator_mode):
    retriever = build(dense_enabled)
    error = None
    try:
        classifier = Stance(retriever)
    except Exception as problem:
        classifier, error = None, f'NLI unavailable ({type(problem).__name__}); verification will abstain.'
    return Pipeline(retriever, Generator(generator_mode), classifier), error

with st.sidebar:
    st.header('Run configuration')
    method = st.selectbox('Retrieval', ['hybrid', 'bm25', 'tfidf', 'dense'])
    generator_mode = st.selectbox('Answer generator', ['local', 'extractive', 'remote'])
    claim_mode = st.checkbox('Audit an atomic claim directly', value=False)
    k = st.slider('Top-K sources', 3, 10, 5)
    st.write(f'NLI gate: {NLI_THRESHOLD}; query cosine gate: 0.30; stability gate: {STABILITY_THRESHOLD}')
    st.caption('Distinct IDs are not proof of independence. Exact duplicate abstracts count as one source group.')

query = st.text_input('Question or claim', 'What potential does antiretroviral therapy have to prevent HIV-associated tuberculosis?')
rewrites_text = st.text_area('Optional reviewed equivalent queries (one per line)', height=80,
                            help='Blank uses template wording probes, not verified paraphrases.')
if st.button('Retrieve, answer, and stress-test', type='primary'):
    try:
        with st.spinner('Retrieving and auditing real corpus evidence… first run loads models.'):
            pipeline, problem = resources(method in ('hybrid', 'dense'), generator_mode)
            rewrites = [line.strip() for line in rewrites_text.splitlines() if line.strip()] or None
            result = pipeline.run(query, method, k, claim_mode, rewrites)
            st.session_state['result'] = result
            st.session_state['problem'] = problem
    except Exception as problem:
        st.error(f'Run failed ({type(problem).__name__}): {problem}. Check dataset/model setup in README.')

def show_hits(hits):
    for hit in hits:
        with st.expander(f"#{hit['rank']} · [{hit['source_id']}] {hit['title']}"):
            st.write({key: value for key, value in hit.items() if key not in ('title', 'passage')})
            st.write(hit['passage'])

if 'result' in st.session_state:
    result = st.session_state['result']
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader(f"Final decision: {result['decision']}")
        st.write(result['final_answer'])
        if result.get('answer_repair'):
            with st.expander('Why the answer was corrected / exact replacement evidence'):
                st.json(result['answer_repair'])
                st.write('Rejected generated claims:')
                st.json(result['superseded_claims'], expanded=False)
        st.caption(f"Generation: {result['generation_mode']} · {result['elapsed_seconds']:.1f}s · retrieval: {result['method']}")
        if result['warning']:
            st.warning(result['warning'])
        if st.session_state.get('problem'):
            st.warning(st.session_state['problem'])
        with st.expander('Initial answer before stress-testing'):
            st.write(result['initial_answer'])
        st.subheader('Evidence stress tests')
        st.caption(result['claim_extraction'])
        for index, audit in enumerate(result['claims'], 1):
            st.markdown(f"**Claim {index}: {audit['claim']}**")
            st.write(f"{audit['decision']} / {audit['status']}: {audit['reason']}")
            st.json(audit['components'], expanded=True)
            with st.expander('Claim evidence: quotes, stance scores and source IDs'):
                st.dataframe(audit['evidence'], width='stretch')
            with st.expander('Counter-evidence queries and actual retrieved passages'):
                for run in audit['counter_evidence']['runs']:
                    st.write(run['query'])
                    show_hits(run['hits'])
            with st.expander('Source removal: what was excluded and what survived'):
                if not audit['ablation']['tests']:
                    st.write('No supporting source qualified for an ablation test.')
                for test in audit['ablation']['tests']:
                    st.write(f"Removed {test['excluded_source_ids']}; support survived: {test['survived']}")
                    show_hits(test['reretrieved'])
                    st.dataframe(test['evidence'], width='stretch')
        if result['decision'] == 'ABSTAIN':
            st.warning('Why the system abstained: ' + '; '.join(dict.fromkeys(a['reason'] for a in result['claims'])))
        with st.expander('Query wording stability'):
            st.write(result['stability']['kind'])
            st.json(result['stability'])
    with right:
        st.subheader('Ranked evidence inspector')
        show_hits(result['retrieved'])
        with st.expander('IR dictionary, DF, IDF and postings sample'):
            st.json(result['query_terms'])
        st.download_button('Download complete audit JSON', json.dumps(result, indent=2), 'ragstress-audit.json', 'application/json')

with st.expander('Measured retrieval evaluation'):
    path = RESULTS / 'retrieval-test.json'
    if path.exists():
        evaluation = json.loads(path.read_text(encoding='utf8'))
        st.dataframe(evaluation['summary'], width='stretch')
        st.caption(evaluation['protocol'])
        image = RESULTS / 'retrieval-test.png'
        if image.exists():
            st.image(str(image))
    else:
        st.info('Run python -m src.cli evaluate to produce actual results.')
