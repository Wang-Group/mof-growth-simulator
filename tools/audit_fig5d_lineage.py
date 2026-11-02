"""Replay historical Fig. 5D assignments against frozen representatives, offline.

This audits the original radial-histogram classifier, not graph isomorphism.
Archived notebooks, pickle files and production KMC are never executed.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

import numpy as np

from tools.audit_fig5_sources import conditions, matching_rows

ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / '15_fig5d_trajectory_lineage'
TABLE = ROOT / '04_btb_motifs/data/unique_structures_result.csv'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def descriptor(raw):
    lines = raw.decode('utf-8').splitlines()
    start = lines.index('@<TRIPOS>ATOM') + 1
    atoms = []
    for line in lines[start:]:
        if line.startswith('@<TRIPOS>'):
            break
        fields = line.split()
        if fields:
            atoms.append((fields[1], *map(float, fields[2:5])))
    if not atoms:
        raise ValueError('Empty MOL2 atom block')
    counts = Counter(a[0] for a in atoms)
    sbus = int(counts['Zr'] / 6)
    linkers = int((counts['C'] - 12 * sbus) / 27)
    # The historical classifier counts first, then removes duplicate atom rows.
    atoms = list(dict.fromkeys(atoms))
    elems = np.array([a[0] for a in atoms])
    xyz = np.array([a[1:] for a in atoms], dtype=float)
    zr = elems == 'Zr'
    center = xyz[zr].mean(axis=0) if zr.any() else xyz.mean(axis=0)
    distances = np.linalg.norm(xyz - center, axis=1)
    hist = {}
    for element in np.unique(elems):
        values = np.histogram(distances[elems == element], bins=np.linspace(0, 50, 51))[0].astype(float)
        hist[element] = values / values.sum() if values.sum() > 0 else values
    return sbus, linkers, hist


def similarity(a, b):
    if not a and not b:
        return 0.0
    scores = []
    for element in sorted(set(a) | set(b)):
        x, y = a.get(element, np.zeros(50)), b.get(element, np.zeros(50))
        scores.append(np.dot(x, y) / ((np.linalg.norm(x) + 1e-12) * (np.linalg.norm(y) + 1e-12)))
    return float(np.mean(scores))


def first_matching_uid(value, representatives):
    for uid, other in sorted(representatives.items()):
        if value[:2] == other[:2] and similarity(value[2], other[2]) > 0.8:
            return uid
    return None


def raw_location(level_index):
    block = int(level_index) - 1
    if block < 0:
        raise ValueError('No preceding saved block')
    file_id = 0 if block == 0 else 1 + (block - 1) // 10000
    first = 0 if file_id == 0 else 1 + 10000 * (file_id - 1)
    return block, f'all_molecules_{file_id}.0.mol2', block - first


def read_csv(path):
    return list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8'))))


def verify(full_source_archive=None):
    manifest = json.loads((GROUP / 'manifest.json').read_text(encoding='utf-8'))
    assert digest(TABLE.read_bytes()) == manifest['classification_table_sha256']
    for item in manifest['files']:
        path = (GROUP / item['path']).resolve()
        assert path.is_relative_to(GROUP.resolve())
        raw = path.read_bytes()
        assert len(raw) == item['bytes'] and digest(raw) == item['sha256'], item['path']
    payload = {}
    for archive_name in manifest['archives']:
        with zipfile.ZipFile(GROUP / archive_name) as archive:
            for member in archive.infolist():
                name = PurePosixPath(member.filename)
                assert not name.is_absolute() and '..' not in name.parts
                assert member.file_size < 2_000_000 and not member.is_dir()
                key = (archive_name, member.filename)
                assert key not in payload
                payload[key] = archive.read(member)
    sources = manifest['sources']
    for item in sources:
        raw = payload[item['archive'], item['member']]
        assert len(raw) == item['bytes'] and digest(raw) == item['sha256']
    assert len(payload) == len(sources), 'Unindexed archive member'
    rows = read_csv(TABLE)
    rowmap = read_csv(GROUP / 'row_to_run.csv')
    snapshots = read_csv(GROUP / 'snapshot_assignments.csv')
    runtime = json.loads((GROUP / 'remote_lineage.json').read_text(encoding='utf-8'))
    raw_runs = {r['folder']: r for r in runtime['runs']}
    assert len(rows) == len(rowmap) == len(raw_runs) == 284
    source_hashes = {s['sha256'] for s in sources}
    notebook = json.loads(payload['data/metadata.zip', 'source/classification_test.ipynb'])
    assert '"block" in file' in ''.join(notebook['cells'][8]['source'])
    for item in manifest['notebooks']:
        from tools.package_entropy_sources import cell_digest
        nb = json.loads(payload[item['archive'], item['member']])
        assert cell_digest(nb) == item['cell_sources_sha256']
        assert all(not c.get('outputs') and c.get('execution_count') is None for c in nb['cells'])
    representatives = {}
    for item in sources:
        if item['kind'] == 'representative':
            p = PurePosixPath(item['member'])
            n, uid = int(p.parent.name.split('_')[-1]), int(p.stem.split('_')[-1])
            representatives.setdefault(n, {})[uid] = descriptor(payload[item['archive'], item['member']])
    assert sum(map(len, representatives.values())) == 352
    existing = 0
    for p in (ROOT / '04_btb_motifs/structures').rglob('*.mol2'):
        assert p.read_bytes() == payload['data/representatives.zip', f'{p.parent.name}/{p.name}']
        existing += 1
    assert existing == 90
    by_cell = {(int(s['row_id']), int(s['entity'])): s for s in snapshots}
    assert len(by_cell) == len(snapshots) == 7978
    full = {}
    if full_source_archive:
        # Optional full recovery archive stays external to the lightweight Git payload.
        fingerprint = manifest['external_full_recovery_archive']
        assert Path(full_source_archive).stat().st_size == fingerprint['bytes']
        with Path(full_source_archive).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == fingerprint['sha256']
        with tarfile.open(full_source_archive, 'r:gz') as archive:
            for member in archive:
                if '/block_' in member.name and member.name.endswith('.mol2'):
                    assert member.isfile() and member.size < 2_000_000
                    assert member.name not in full
                    full[member.name] = archive.extractfile(member).read()
        assert len(full) == 7978
    verified = 0
    figure_cells = 0
    for number, mapping in enumerate(rowmap, 1):
        assert int(mapping['row_id']) == number
        assert matching_rows(conditions(mapping['folder']), rows) == [number]
        run = raw_runs[mapping['folder']]
        assert run['raw_folder_exists'] and all(c['exists'] for c in run['raw_chunks'])
        assert all(c['exists'] and c['sha256'] in source_hashes for c in run['code'])
        levels = json.loads(payload['data/metadata.zip', f'indices/r{number:03d}.json'])
        for entity in range(1, 30):
            expected = rows[number-1][f'entity_{entity}']
            if not expected:
                assert (number, entity) not in by_cell
                continue
            item = by_cell[number, entity]
            assert int(float(expected)) == int(item['uid']) == int(item['first_matching_uid'])
            info = levels[str(entity)]
            block, chunk, offset = raw_location(info['idx'])
            assert block == int(item['block_index']) and chunk == item['raw_chunk']
            assert offset == int(item['local_block_index'])
            assert float(item['raw_time_u']) == float(info['x_value'])
            assert int(item['level_idx']) == int(info['idx'])
            raw = None
            if entity <= 13:
                figure_cells += 1
                raw = payload[item['archive'], item['member']]
            if full:
                recovered = full[mapping['folder'] + f'/block_{block}.mol2']
                assert raw is None or raw == recovered
                raw = recovered
            if raw is not None:
                assert len(raw) == int(item['bytes']) and digest(raw) == item['sha256']
                value = descriptor(raw)
                assert value[0] + value[1] == entity
                assert value[:2] == (int(item['sbu_count']), int(item['btb_count']))
                assert first_matching_uid(value, representatives[entity]) == int(item['uid'])
                score = similarity(value[2], representatives[entity][int(item['uid'])][2])
                assert np.isclose(score, float(item['similarity_to_recorded_uid']), rtol=0, atol=1e-12)
                verified += 1
    assert figure_cells == 3688 and verified == (7978 if full else 3688)
    original_manifest = json.loads((ROOT / '14_fig5_source_followup/source_manifest.json').read_text())
    external = [x for x in original_manifest['files'] if x['kind'] == 'external_trajectory']
    for item in external:
        matches = [r for r in runtime['trace_hashes'] if r['sha256'] == item['source_sha256']]
        assert len(matches) == 1
        assert conditions(matches[0]['path'])['Index'] == item['source_relative'].split('_index')[-1].split('.')[0]
    checks = json.loads((GROUP / 'raw_block_checks.json').read_text())['checks']
    assert len(checks) == 4 and all(r['matches'] and r['snapshot_sha256'] == r['raw_block_sha256'] for r in checks)
    for check in checks:
        number = matching_rows(conditions(check['folder']), rows)[0]
        assert by_cell[number, 13]['sha256'] == check['raw_block_sha256']
    return {'status': 'PASS_FIG5D_LINEAGE', 'classification_rows_linked': 284,
            'indexed_classification_cells': 7978, 'replayed_cells_this_run': verified,
            'packaged_fig5d_snapshots': figure_cells, 'representatives_byte_matched_to_group04': existing,
            'original_raw_block_hash_spot_checks': 4, 'resolved_external_trajectory_indices': 3,
            'scope': 'Frozen-representative radial-histogram replay and recorded provenance. Not a new classifier, graph-isomorphism proof, RNG replay, or full raw-stream re-extraction.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-source-archive', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.full_source_archive), indent=2))
