"""Verify archived BTB structural-analysis source files and reference hashes."""
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
from tools.data_retention import excluded_payload

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


def verify():
    manifest = json.loads((GROUP / 'source_manifest.json').read_text(encoding='utf-8'))
    if digest(TABLE.read_bytes()) != manifest['classification_table_sha256']:
        raise AssertionError('Classification table changed')
    verified_sources = []
    for item in manifest['files']:
        if not item.get('release_path'):
            if item['kind'] != 'external_trajectory' or len(item['source_sha256']) != 64:
                raise AssertionError('Invalid external evidence')
            continue
        path = (ROOT / item['release_path']).resolve()
        if not path.is_relative_to(GROUP.resolve()):
            raise AssertionError('Source outside group')
        if excluded_payload(path, item['stored_bytes'], item['stored_sha256']):
            continue
        stored = path.read_bytes()
        if len(stored) != item['stored_bytes'] or digest(stored) != item['stored_sha256']:
            raise AssertionError('Stored source changed')
        verified_sources.append(item['release_path'])
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
    if len(rows) != 284 or len(verified_sources) != 2:
        raise AssertionError('Unexpected source inventory')
    return {'status': 'PASS', 'classification_rows': len(rows),
            'verified_source_files': verified_sources,
            'trajectory_mapping_entry': 'tools/audit_fig5d_lineage.py'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
