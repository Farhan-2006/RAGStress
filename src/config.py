from pathlib import Path
import os

# Use OS trust anchors (especially Windows enterprise roots), never disable TLS verification.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
CACHE = ROOT / 'cache'
RESULTS = ROOT / 'results'
for directory in (DATA, CACHE, RESULTS):
    directory.mkdir(exist_ok=True)
os.environ.setdefault('HF_HOME', str(CACHE / 'huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
DENSE_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
NLI_MODEL = os.environ.get('RAG_NLI_MODEL', 'cross-encoder/nli-deberta-v3-small')
GENERATOR_MODEL = 'google/flan-t5-small'
MODEL_REVISIONS = {
    DENSE_MODEL: '1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
    'cross-encoder/nli-MiniLM2-L6-H768': 'b95119ce93d3e065de6214e38cd4a97b0f2f2c6d',
    'cross-encoder/nli-deberta-v3-small': 'fa2804872c3b4bd748f38c0185cc85775361e735',
    GENERATOR_MODEL: '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab',
}
SEED = 358
TOP_K = 5
POOL_K = 50
RRF_CONSTANT = 60
# Prototype thresholds, frozen before held-out evaluation; not probabilities of truth.
NLI_THRESHOLD = 0.80
STABILITY_THRESHOLD = 0.40
