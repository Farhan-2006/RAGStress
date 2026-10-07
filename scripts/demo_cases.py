"""Live demonstrations; saved audits never drive app responses."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cli import build, write_json, write_csv
from src.generator import Generator
from src.stance import Stance
from src.pipeline import Pipeline
from src.config import RESULTS

retriever = build()
pipeline = Pipeline(retriever, Generator('local'), Stance(retriever))
cases = [
    ('supported', 'Does antiretroviral therapy reduce tuberculosis incidence?', False),
    ('claim-support', 'Antiretroviral therapy has substantial potential to prevent HIV-associated tuberculosis.', True),
    ('domain-limitation', 'Activation of PPM1D suppresses p53 function.', True),
    ('counter-claim', 'Antiretroviral therapy has no potential to prevent HIV-associated tuberculosis.', True),
    ('contradiction', "Transplanted human glial progenitor cells are incapable of forming a neural network with host animals' neurons.", True),
    ('unanswerable', 'What is the certified wingspan of the invisible blue dragon on planet Zorblax-99?', False),
]
rows = []
for label, query, claim_mode in cases:
    result = pipeline.run(query, claim_mode=claim_mode)
    write_json(RESULTS / f'demo-{label}.json', result)
    rows.append({'case': label, 'query': query, 'decision': result['decision'],
                 'generation_mode': result['generation_mode'], 'claims': len(result['claims']),
                 'runtime_seconds': result['elapsed_seconds'],
                 'ablation_tests': sum(len(c['ablation']['tests']) for c in result['claims']),
                 'final_answer': result['final_answer']})
    print(label, result['decision'], result['final_answer'], flush=True)
write_csv(RESULTS / 'demo-cases.csv', rows)
write_json(RESULTS / 'demo-cases.json', rows)
