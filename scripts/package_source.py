"""Package original project code and measured results; exclude envs/data/weights/secrets."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
output = root.parent / 'RAGStress_Source.zip'
excluded = {'.git', '.venv', 'cache', 'data', '__pycache__', '.pytest_cache'}
with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if path.is_file() and not any(part in excluded for part in relative.parts) and path.name != '.env' and not path.name.endswith('.log'):
            archive.write(path, Path('ragstress') / relative)
    manifest = root / 'data/manifest.json'
    if manifest.exists() and not (root / 'dataset-manifest.json').exists():
        archive.write(manifest, 'ragstress/dataset-manifest.json')
print(output, output.stat().st_size)
