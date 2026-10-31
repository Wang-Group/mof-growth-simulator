"""Production bookkeeping only. Molecular event rules remain frozen."""
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_campaign(path, verify=True):
    path = Path(path).resolve()
    m = json.loads(path.read_text())
    base = path.parent
    if verify:
        for name, digest in m['files'].items():
            if sha(base / name) != digest:
                raise ValueError('Frozen campaign file changed: ' + name)
    return base, m


def setup(base, manifest):
    integration = base / 'frozen/integration'
    sys.path.insert(0, str(integration))
    from runtime import configure
    bundle = base / 'frozen/bundle'
    root = bundle / 'steric_mixed_solvent_20260912_v1'
    engine, sources = configure(root, bundle, integration)
    provenance = {'manifest_sha256': sha(base / 'manifest.json'),
                  'engine_sources': sources, 'frozen_files': manifest['files']}
    return engine, root, base / 'frozen/acetate.mol', provenance


def lightweight(sim):
    """No vacancy reconstruction and no random draws during growth."""
    m = sim.model
    prop = m.propensities()
    ext = prop['add_node'] + prop['add_linker']
    return {'events': sim.event_count, 'tau': sim.tau,
            'N': len(m.assembly.entities), 'maximum_N': sim.maximum_N,
            'pool': m.ledger.record(), 'D_count': len(m.intermediates),
            'bonds': len(m.assembly.linked_carboxylate_pairs),
            'cap_count': len(m.cap), 'solvent_cap_count': m.solvent_count(),
            'proposal_propensities': prop,
            'total_removal_external_addition_proposal_ratio':
                prop['remove_bond'] / ext if ext else None}
