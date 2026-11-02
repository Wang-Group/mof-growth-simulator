"""Curate an already downloaded, read-only Jiageng audit into a lightweight group.

This importer is optional. Normal offline verification needs no remote access.
It refuses changed outputs and never deserializes pickle or executes notebooks.
"""
import argparse
import base64
import csv
import io
import json
from pathlib import Path
import tarfile
import zipfile

from tools.audit_fig5d_lineage import GROUP, TABLE, digest, raw_location
from tools.package_entropy_sources import clean_notebook, write_new_or_identical


def json_bytes(value):
    return (json.dumps(value, indent=2) + '\n').encode('utf-8')


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-dir', type=Path, required=True)
    args = parser.parse_args()
    recovery = args.recovery_dir.resolve(strict=True)
    audit = json.loads((recovery/'classification_audit.json').read_text())
    lineage = json.loads((recovery/'lineage.json').read_text())
    inventory = json.loads((recovery/'inventory.json').read_text())
    assert audit['classification_mismatches'] == [] and audit['snapshots'] == 7978
    expected = {r['path']: r['bytes'] for r in inventory['files']}
    payload = {}
    with tarfile.open(recovery/'classified_sources.tar.gz', 'r:gz') as archive:
        for member in archive:
            assert member.isfile() and member.name in expected and member.name not in payload
            assert member.size == expected[member.name] < 2_000_000
            payload[member.name] = archive.extractfile(member).read()
    assert set(payload) == set(expected)
    assert payload['unique_structures_result.csv'] == TABLE.read_bytes()
    sources, notebooks, bundles = [], [], {}

    def add(archive, member, raw, original, kind, notebook=False):
        stored, cells = clean_notebook(raw) if notebook else (raw, None)
        assert member not in bundles.setdefault(archive, {})
        bundles[archive][member] = stored
        record = {'archive': archive, 'member': member, 'kind': kind,
                  'source_path': original, 'bytes': len(stored), 'sha256': digest(stored),
                  'source_bytes': len(raw), 'source_sha256': digest(raw),
                  'transformation': 'notebook_outputs_metadata_cleared' if notebook else 'byte_identical'}
        sources.append(record)
        if notebook:
            notebooks.append({**record, 'cell_sources_sha256': cells})

    metadata_zip = 'data/metadata.zip'
    processed = lineage['processed_root']
    for mapping in audit['rowmap']:
        number, folder = mapping['row_id'], mapping['folder']
        name = folder + '/level_idx.json'
        add(metadata_zip, f'indices/r{number:03d}.json', payload[name], processed+'/'+name, 'level_index')
    for name, raw in sorted(payload.items()):
        if name.startswith('unique_structures/'):
            add('data/representatives.zip', name.removeprefix('unique_structures/'), raw,
                processed+'/'+name, 'representative')
    add(metadata_zip, 'source/classification_test.ipynb', payload['test.ipynb'],
        processed+'/test.ipynb', 'classifier', notebook=True)
    for sha, item in sorted(lineage['sources'].items()):
        raw = base64.b64decode(item['base64'])
        assert digest(raw) == sha and len(raw) == item['bytes']
        name = Path(item['path']).name
        member = 'source/'+name if name.endswith('.ipynb') else f'run_sources/{sha}/{name}'
        add(metadata_zip, member, raw, item['path'], 'extractor' if name.endswith('.ipynb') else 'run_source',
            notebook=name.endswith('.ipynb'))
    snapshot_rows = []
    for item in audit['entries']:
        record = dict(item)
        block, chunk, offset = raw_location(item['level_idx'])
        raw = payload[item['snapshot']]
        assert digest(raw) == item['sha256'] and block == item['block_index']
        record.update(raw_chunk=chunk, local_block_index=offset, archive='', member='')
        if item['entity'] <= 13:
            # Two small archives keep every Git object below 25 MB.
            partition = '01_07' if item['entity'] <= 7 else '08_13'
            archive = f'data/snapshots_entity{partition}.zip'
            member = f'r{item["row_id"]:03d}/entity_{item["entity"]:02d}.mol2'
            add(archive, member, raw, processed+'/'+item['snapshot'], 'snapshot')
            record.update(archive=archive, member=member)
        del record['snapshot']
        snapshot_rows.append(record)
    plain = {
        'row_to_run.csv': csv_bytes(audit['rowmap']),
        'snapshot_assignments.csv': csv_bytes(snapshot_rows),
        'remote_lineage.json': json_bytes({k: v for k, v in lineage.items() if k != 'sources'}),
        'raw_block_checks.json': (recovery/'raw_block_checks.json').read_bytes(),
        'full_recovery_audit.json': json_bytes({k: v for k, v in audit.items() if k not in {'rowmap', 'entries'}}),
    }
    records = []
    for name, members in bundles.items():
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for member, raw in sorted(members.items()):
                info = zipfile.ZipInfo(member, (2026, 11, 2, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        plain[name] = stream.getvalue()
    for name, raw in sorted(plain.items()):
        assert len(raw) < 25_000_000, (name, len(raw))
        write_new_or_identical(GROUP/name, raw)
        records.append({'path': name, 'bytes': len(raw), 'sha256': digest(raw)})
    manifest = {'recovered_on': '2026-11-02', 'classification_table_sha256': digest(TABLE.read_bytes()),
                'processed_root': processed, 'raw_root': lineage['raw_root'],
                'files': records, 'archives': sorted(bundles), 'sources': sources, 'notebooks': notebooks,
                'external_full_recovery_archive': {'bytes': (recovery/'classified_sources.tar.gz').stat().st_size,
                                                   'sha256': digest((recovery/'classified_sources.tar.gz').read_bytes())},
                'scope': 'All 284 rows indexed; full imported cohort replay checked 7978 assignments. '
                         'Git carries 3688 entity 1-13 snapshots; entity 14-29 snapshots remain external. '
                         'Four raw blocks hash-checked, not every raw event re-extracted.'}
    write_new_or_identical(GROUP/'manifest.json', json_bytes(manifest))
    print(json.dumps({'files': len(records), 'source_members': len(sources),
                      'payload_bytes': sum(r['bytes'] for r in records),
                      'archive_bytes': {r['path']: r['bytes'] for r in records if r['path'].endswith('.zip')}}, indent=2))


if __name__ == '__main__':
    main()
