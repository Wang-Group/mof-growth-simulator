"""Import the reviewed NAS entropy subset without executing its code."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / '08_entropy_inputs/source'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def cell_digest(nb):
    return digest(json.dumps([(c['cell_type'], c['source']) for c in nb['cells']],
                             ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


def clean_notebook(raw):
    original = json.loads(raw)
    cleaned = {'cells': [], 'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5}
    for i, cell in enumerate(original['cells']):
        item = {'cell_type': cell['cell_type'], 'id': f'source-{i}',
                'metadata': {}, 'source': cell['source']}
        if cell['cell_type'] == 'code':
            item.update(execution_count=None, outputs=[])
        cleaned['cells'].append(item)
    assert cell_digest(original) == cell_digest(cleaned)
    return (json.dumps(cleaned, ensure_ascii=False, indent=1) + '\n').encode('utf-8'), cell_digest(original)


def write_new_or_identical(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f'Refusing to replace changed file: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    selected = sorted(p for p in source.rglob('*') if p.is_file()
                      and p.suffix in {'.py', '.ipynb', '.mol', '.xyz'}
                      and p.parent in {source, source / 'mol_xyz_files'})
    if len(selected) != 39:
        raise ValueError(f'Expected the reviewed 39-file subset, found {len(selected)}')
    records = []
    for path in selected:
        relative = path.relative_to(source)
        raw = path.read_bytes()
        stored, cells = clean_notebook(raw) if path.suffix == '.ipynb' else (raw, None)
        target = DEST / relative
        write_new_or_identical(target, stored)
        records.append({'source_relative': relative.as_posix(),
                        'release_path': target.relative_to(ROOT).as_posix(),
                        'source_bytes': len(raw), 'source_sha256': digest(raw),
                        'stored_bytes': len(stored), 'stored_sha256': digest(stored),
                        'transformation': 'notebook_outputs_metadata_cleared' if cells else 'byte_identical',
                        'cell_sources_sha256': cells})
    for record in records:
        assert digest((source / record['source_relative']).read_bytes()) == record['source_sha256']
    manifest = {'source_relative_to_shared_project': 'MOF_growth_Data/Entropy calculation/entropy',
                'imported_on': '2026-11-01',
                'scope': 'All six reviewed code/notebook files and all 33 supplied MOL/XYZ geometries. Original code cells retained; notebook outputs and metadata omitted. Existing data tables are in ../data.',
                'files': records}
    write_new_or_identical(ROOT / '08_entropy_inputs/calculation_manifest.json',
                          (json.dumps(manifest, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'files': len(records), 'stored_bytes': sum(r['stored_bytes'] for r in records)}))


if __name__ == '__main__':
    main()
