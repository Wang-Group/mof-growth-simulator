"""Enforce the compact publication payload without weakening source checks."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def policy():
    return json.loads((ROOT / 'metadata/data_retention.json').read_text(encoding='utf-8'))


def excluded_payload(path, expected_bytes, expected_hash):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Payload outside repository')
    relative = path.relative_to(ROOT.resolve()).as_posix()
    for row in policy()['removed_payloads']:
        if row['path'] == relative:
            if row['bytes'] != expected_bytes or row['sha256'] != expected_hash:
                raise AssertionError('Exclusion does not match historical source')
            if path.exists():
                raise AssertionError(f'Excluded raw payload reintroduced: {relative}')
            return True
    return False


def classifier_examples():
    result = {}
    for row in policy()['classifier_fixtures']:
        path = (ROOT / row['path']).resolve()
        if not path.is_relative_to(ROOT / '15_fig5d_trajectory_lineage/examples'):
            raise AssertionError('Example outside fixture directory')
        raw = path.read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise AssertionError('Changed classifier example')
        key = (row['archive'], row['member'])
        if key in result:
            raise AssertionError('Duplicate classifier example')
        result[key] = raw
    if len(result) != 4:
        raise AssertionError('Expected four documented regression examples')
    return result


def verify():
    data = policy()
    removed = data['removed_payloads']
    if len(removed) != 11 or len({r['path'] for r in removed}) != 11:
        raise AssertionError('Unexpected raw-output exclusion inventory')
    for row in removed:
        if not excluded_payload(ROOT / row['path'], row['bytes'], row['sha256']):
            raise AssertionError('Unregistered exclusion')
    examples = classifier_examples()
    return dict(status='PASS_COMPACT_PAYLOAD', excluded_raw_files=len(removed),
                excluded_bytes=sum(r['bytes'] for r in removed),
                classifier_examples=len(examples),
                scope='Current tree only; earlier commits and original research copies are unchanged')
