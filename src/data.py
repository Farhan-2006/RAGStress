"""Download dataset archives only, never an existing project's implementation."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
import zipfile
from .config import DATA

URLS = {
    'beir': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip',
    'original': 'https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz',
}

def download(url, destination):
    destination = Path(destination)
    if destination.exists():
        return
    temporary = destination.with_suffix(destination.suffix + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'RAGStress-academic/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)

def prepare():
    beir_zip, original_tar = DATA / 'scifact.zip', DATA / 'scifact-original.tar.gz'
    download(URLS['beir'], beir_zip)
    # Extract only expected data files; archive paths cannot escape DATA.
    with zipfile.ZipFile(beir_zip) as archive:
        for name in ('scifact/corpus.jsonl', 'scifact/queries.jsonl',
                     'scifact/qrels/train.tsv', 'scifact/qrels/test.tsv'):
            target = DATA / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    download(URLS['original'], original_tar)
    original = DATA / 'original'
    original.mkdir(exist_ok=True)
    with tarfile.open(original_tar) as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if name in ('corpus.jsonl', 'claims_train.jsonl', 'claims_dev.jsonl', 'claims_test.jsonl') and member.isfile():
                stream = archive.extractfile(member)
                (original / name).write_bytes(stream.read())
    manifest = {key: {'url': URLS[key], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                for key, path in [('beir', beir_zip), ('original', original_tar)]}
    (DATA / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    return manifest

def jsonl(path):
    with Path(path).open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def load_corpus():
    path = DATA / 'scifact/corpus.jsonl'
    if not path.exists():
        raise FileNotFoundError('Run python -m src.cli download first.')
    return {str(row['_id']): {'id': str(row['_id']), 'title': row['title'], 'text': row['text']}
            for row in jsonl(path)}

def load_queries():
    return {str(row['_id']): row['text'] for row in jsonl(DATA / 'scifact/queries.jsonl')}

def load_qrels(split='test'):
    qrels = {}
    with (DATA / f'scifact/qrels/{split}.tsv').open(encoding='utf8') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            qrels.setdefault(row['query-id'], {})[row['corpus-id']] = int(row['score'])
    return qrels
