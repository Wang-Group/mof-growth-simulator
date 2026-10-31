"""Import audited source bytes; never overwrite a different packaged version."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wave-audit', required=True, type=Path)
    parser.add_argument('--shared-audit', required=True, type=Path)
    parser.add_argument('--shared-root', required=True, type=Path)
    args = parser.parse_args()
    records = []

    def copy(source, destination, origin, expected=None):
        target = (ROOT / destination).resolve()
        if not target.is_relative_to(ROOT) or target.suffix.lower() in {'.docx', '.pkl', '.pickle', '.pem', '.key'}:
            raise ValueError(f'Invalid release path: {destination}')
        data = source.read_bytes()
        if len(data) > 25_000_000:
            raise ValueError(f'Large file requires external archive: {source}')
        fingerprint = sha(data)
        if expected and fingerprint != expected:
            raise ValueError(f'Source changed since audit: {source}')
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f'Refusing to overwrite another version: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if sha(target.read_bytes()) != fingerprint:
            raise ValueError(f'Copy verification failed: {target}')
        records.append({'release_path': target.relative_to(ROOT).as_posix(),
                        'bytes': len(data), 'sha256': fingerprint,
                        'operation': 'byte-identical copy', 'origin': origin})

    wave = json.loads((args.wave_audit / 'local_verification.json').read_text(encoding='utf-8-sig'))
    if wave['status'] != 'PASS' or len(wave['files']) != 10:
        raise ValueError('Unexpected wave audit')
    for row in wave['files']:
        copy(args.wave_audit / 'retrieved_5090' / row['path'], '09_recovered_runtime/' + row['path'],
             {'kind': 'authorized read-only SSH export', 'host_alias': '5090',
              'project_path': '/home/wave/MOL_growth_recalculation_20260830/' + row['path'],
              'audit_date': '2026-10-31'}, row['sha256'])

    inventory = json.loads((args.shared_audit / 'source_inventory.json').read_text(encoding='utf-8-sig'))
    indexed = {r['relative_path'].replace('\\', '/'): r for r in inventory['files']}
    prefix = 'MOF_growth_Data/MOL_figure_source_bundles_20261022_v4/'
    bundle = args.shared_root / prefix
    versions = [
        ('SI_Figure_S32/content_checked_20261023/original_code/build_figure7_kinetic_window_v2.py',
         'historical/figure6/build_figure7_kinetic_window_v2.py'),
        ('SI_Figure_S7/content_checked_20261023/backup_raw_u_20261027/panel_runtime.py',
         'historical/common/panel_runtime.py'),
        ('MS_Figure_6/content_checked_20261023/original_code/build_figure7_kinetic_window_v2.py',
         'shared_20261031/MS_Figure_6/original_code/build_figure7_kinetic_window_v2.py'),
        ('MS_Figure_6/content_checked_20261023/panel_runtime.py', 'shared_20261031/MS_Figure_6/panel_runtime.py'),
        ('MS_Figure_6/content_checked_20261023/figure6B_reference_style.py',
         'shared_20261031/MS_Figure_6/figure6B_reference_style.py'),
        ('SI_Figure_S7/content_checked_20261023/panel_runtime.py', 'shared_20261031/SI_Figure_S7/panel_runtime.py'),
        ('SI_Figure_S7/content_checked_20261023/original_code/export_figure_s7_panels.py',
         'shared_20261031/SI_Figure_S7/original_code/export_figure_s7_panels.py'),
        ('SI_Figure_S7/content_checked_20261023/original_code/export_figure_s7_panels_seconds.py',
         'shared_20261031/SI_Figure_S7/original_code/export_figure_s7_panels_seconds.py'),
    ]
    for original, destination in versions:
        copy(bundle / original, '10_plot_versions/' + destination,
             {'kind': 'shared project file', 'source_relative': prefix + original,
              'snapshot_date': '2026-10-31'}, indexed[prefix + original]['sha256'])

    s21 = bundle / 'SI_Figure_S21/April03_source_copy_20261030'
    files = sorted(p for p in s21.rglob('*') if p.is_file())
    if len(files) != 24:
        raise ValueError('S21 source set changed; re-audit before importing')
    for source in files:
        relative = source.relative_to(s21).as_posix()
        copy(source, '11_s21_april03/source_archive/' + relative,
             {'kind': 'shared project file',
              'source_relative': prefix + 'SI_Figure_S21/April03_source_copy_20261030/' + relative,
              'snapshot_date': '2026-10-31',
              'variant': 'initialize_structures1 alternative' if relative.startswith('other_versions/')
                         else 'Zr6_Zr12_mix April03 source package'})

    index_evidence = json.loads((args.wave_audit / 'required_files.json').read_text(encoding='utf-8-sig'))
    remote_index = next(r for r in index_evidence['archive_members'] if r['member'].endswith('/REMOTE_FILE_INDEX.csv'))
    manifest = {'schema_version': 1, 'import_date': '2026-10-31',
                'base_commit': 'd2fd39a47d5deb000e05130fc01c1ef529bbbd7e',
                'index_provenance': {'provided_archive_sha256': index_evidence['archive_sha256'],
                                     'remote_file_index_sha256': remote_index['sha256']},
                'files': records,
                'limits': ['No KMC rerun', 'No MS/SI modification or fresh document matching',
                           'Archived paths are provenance, not runtime requirements']}
    target = ROOT / 'metadata/recovered_sources_20261031.json'
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    ledger = ROOT / 'metadata/publication_sources.csv'
    with ledger.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, existing = reader.fieldnames, list(reader)
    changed = 0
    for row in existing:
        if row['decision'] not in {'HASH_MISMATCH_REVIEW_REQUIRED', 'VERSION_VARIANT_ARCHIVED'}:
            continue
        matches = [r for r in records if r['origin'].get('source_relative') == prefix + row['source_relative']]
        if len(matches) != 1 or matches[0]['sha256'] != row['sha256']:
            raise ValueError(f'Cannot resolve ledger version: {row["source_relative"]}')
        row['release_path'] = matches[0]['release_path']
        row['decision'] = 'VERSION_VARIANT_ARCHIVED'
        row['scope'] = 'Shared revision archived separately; expected_sha256 identifies the historical version retained in 10_plot_versions/historical. See 10_plot_versions/README.md; final artwork adoption pending.'
        changed += 1
    if changed != 3:
        raise ValueError('Expected exactly three version-discrepancy ledger rows')
    with ledger.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)
    print(json.dumps({'files': len(records), 'bytes': sum(r['bytes'] for r in records)}, indent=2))


if __name__ == '__main__':
    main()
