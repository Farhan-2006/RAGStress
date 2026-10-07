"""Render the UI against recorded real sanity outputs; never used by the demo app."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def main():
    cases = json.loads((ROOT / 'calibration/sanity_runs.json').read_text(encoding='utf8'))
    by_query = {case['query']: case['result'] for case in cases}

    class RecordedPipeline:
        def __init__(self, *args):
            pass

        def run(self, query, *args, **kwargs):
            return copy.deepcopy(by_query[query])

    checks = []
    with patch('src.cli.build', return_value=None), patch('src.stance.Stance', return_value=None), \
            patch('src.generator.Generator', return_value=None), patch('src.pipeline.Pipeline', RecordedPipeline):
        for case in cases:
            app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
            if case['result']['claim_mode']:
                app.radio(key='input_mode').set_value('Check a claim').run()
            app.text_input(key='query').set_value(case['query'])
            app.button(key='submit_check').click().run(timeout=30)
            assert not app.exception, [e.message for e in app.exception]
            assert app.session_state['result']['decision'] == case['result']['decision']
            assert len(app.metric) == 3
            assert 'Approved source documents' in [e.value for e in app.subheader]
            rendered = '\n'.join(str(e.value) for e in app.markdown)
            assert 'source ablation' not in rendered.casefold()
            assert 'source removal' not in rendered.casefold()
            assert 'Does retrieval change with wording?' in [e.value for e in app.subheader]
            checks.append({'case': case['case'], 'decision': case['result']['decision'],
                           'exceptions': 0, 'summary_metrics': len(app.metric)})
    output = {'mode': 'UI rendering of recorded real backend sanity runs; not a live performance experiment',
              'verified': True, 'cases': checks}
    (ROOT / 'results/ui-verification.json').write_text(json.dumps(output, indent=2), encoding='utf8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
