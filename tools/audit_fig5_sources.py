"""Verify partial Fig. 5D provenance without unpickling or running old notebooks."""
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / '14_fig5_source_followup'
FIELDS = ('Zr', 'FA', 'L', 'Ratio', 'Step', 'Index', 'SC', 'KC')
PATTERN = re.compile(r'(' + '|'.join(FIELDS) + r')_([^_/\\]+)')
TABLE = ROOT / '04_btb_motifs/data/unique_structures_result.csv'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def conditions(path):
    for segment in str(path).replace('\\', '/').split('/'):
        values = dict(PATTERN.findall(segment))
        if set(FIELDS).issubset(values):
            return values
    return None


def matching_rows(values, rows, ignore_index=False):
    if values is None or not set(FIELDS).issubset(values):
        return []
    matches = []
    for number, row in enumerate(rows, 1):
        good = True
        for key in FIELDS:
            if key == 'Index' and ignore_index:
                continue
            if key == 'Step':
                good &= values[key] == row[key]
            else:
                good &= math.isclose(float(values[key]), float(row[key]),
                                     rel_tol=1e-12, abs_tol=1e-12)
        if good:
            matches.append(number)
    return matches


def build_row_map(records, rows):
    result = []
    for number, row in enumerate(rows, 1):
        structures = [r['id'] for r in records if r['kind'] == 'large_structure'
                      and number in matching_rows(conditions(r['source_member']), rows)]
        traces = [r['id'] for r in records if r['kind'] == 'external_trajectory'
                  and number in matching_rows(conditions(r['source_relative']), rows, ignore_index=True)]
        status = ('STRUCTURE_CONDITIONS_ONLY' if structures else
                  'AMBIGUOUS_TRACE_INDEX' if traces else 'NOT_LOCATED')
        result.append({'row_id': number, **{key: row[key] for key in FIELDS},
                       'structure_evidence': ';'.join(structures),
                       'trajectory_candidates': ';'.join(traces),
                       'status': status, 'classification_snapshot_link_verified': 'false'})
    return result


def verify():
    manifest = json.loads((GROUP / 'source_manifest.json').read_text(encoding='utf-8'))
    if digest(TABLE.read_bytes()) != manifest['classification_table_sha256']:
        raise AssertionError('Classification table changed')
    for item in manifest['files']:
        if not item.get('release_path'):
            if item['kind'] != 'external_trajectory' or len(item['source_sha256']) != 64:
                raise AssertionError('Invalid external evidence')
            continue
        path = (ROOT / item['release_path']).resolve()
        if not path.is_relative_to(GROUP.resolve()):
            raise AssertionError('Source outside group')
        stored = path.read_bytes()
        if len(stored) != item['stored_bytes'] or digest(stored) != item['stored_sha256']:
            raise AssertionError('Stored source changed')
        if item['encoding'] == 'gzip':
            raw = gzip.decompress(stored)
        elif item['encoding'] == 'notebook_outputs_metadata_cleared':
            from tools.package_entropy_sources import cell_digest
            nb = json.loads(stored)
            if cell_digest(nb) != item['cell_sources_sha256']:
                raise AssertionError('Notebook cells changed')
            if any(c.get('outputs') or c.get('execution_count') is not None for c in nb['cells']):
                raise AssertionError('Notebook outputs retained')
            continue
        elif item['encoding'] == 'byte_identical':
            raw = stored
        else:
            raise AssertionError('Unknown encoding')
        if len(raw) != item['source_bytes'] or digest(raw) != item['source_sha256']:
            raise AssertionError('Original source changed')
    rows = list(csv.DictReader(io.StringIO(TABLE.read_text(encoding='utf-8'))))
    derived = build_row_map(manifest['files'], rows)
    saved = list(csv.DictReader(io.StringIO((GROUP / 'row_source_map.csv').read_text(encoding='utf-8'))))
    normalized = [{k: str(v) for k, v in row.items()} for row in derived]
    if normalized != saved:
        raise AssertionError('Row mapping differs from packaged evidence')
    counts = {label: sum(row['status'] == label for row in derived) for label in
              ('STRUCTURE_CONDITIONS_ONLY', 'AMBIGUOUS_TRACE_INDEX', 'NOT_LOCATED')}
    if len(rows) != 284 or counts != {'STRUCTURE_CONDITIONS_ONLY': 8, 'AMBIGUOUS_TRACE_INDEX': 4, 'NOT_LOCATED': 272}:
        raise AssertionError('Unexpected evidence coverage')
    structures = [item for item in manifest['files'] if item['kind'] == 'large_structure']
    external = [item for item in manifest['files'] if item['kind'] == 'external_trajectory']
    incomplete = sum(conditions(item['source_member']) is None for item in structures)
    if (len(structures), len(external), incomplete) != (9, 3, 1):
        raise AssertionError('Unexpected source inventory')
    for item in manifest['files']:
        if item['kind'] == 'external_trajectory':
            parent_index = int(float(conditions(item['source_relative'])['Index']))
            filename_index = int(re.search(r'_index(\d+)\.pkl$', item['source_relative']).group(1))
            if parent_index == filename_index:
                raise AssertionError('Expected explicitly unresolved index conflict')
    return {'status': 'PASS_PARTIAL_EVIDENCE_CHECKS', 'classification_rows': len(rows),
            'row_status_counts': counts, 'fully_linked_classification_rows': 0,
            'packaged_large_structures': len(structures), 'incomplete_structure_name_count': incomplete,
            'external_trajectory_hash_records': len(external), 'fig5c_original_module_found': False,
            'scope': 'Offline source/parameter matching only. No classification rerun, no pickle deserialization, no claim that large structures are the classified small-entity snapshots.'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
