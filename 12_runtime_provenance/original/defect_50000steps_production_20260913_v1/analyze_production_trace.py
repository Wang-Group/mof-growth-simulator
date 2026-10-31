"""Read-only central-node filling analysis of all saved size milestones."""
import argparse
import csv
import importlib
import json
from pathlib import Path
import pickle
import sys
import traceback
from campaign_io import load_campaign, setup, sha, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--trace', type=Path, required=True)
    args = ap.parse_args()
    base, manifest = load_campaign(args.manifest)
    # Real SciPy is used ONLY for this separate, read-only measurement process.
    sys.path.insert(0, str(base/'frozen/analysis'))
    from central_filling import CentralLattice, FilledEnvelope, analyze
    engine, root, acetate, provenance = setup(base, manifest)
    for module in ('numeric', 'multiarray', 'umath', '_multiarray_umath'):
        sys.modules.setdefault('numpy._core.'+module, importlib.import_module('numpy.core.'+module))
    from vacancy_counter import assembly_coordinates
    with (root/'framework_inputs/UiO66_N76_reviewed.pkl').open('rb') as f:
        reference = pickle.load(f)
    out = args.trace/'filling_analysis'
    out.mkdir(exist_ok=False)
    records = []
    for path in sorted((args.trace/'snapshots').glob('*/checkpoint.pkl')):
        source_sha = sha(path)
        record = {'snapshot': path.parent.name, 'checkpoint_sha256': source_sha}
        try:
            with path.open('rb') as f:
                saved = pickle.load(f)
            # Read-only measurement deliberately uses a different SciPy runtime.
            # Require every source/input/hash to match, not environment shim notes.
            old_provenance = dict(saved['provenance'])
            new_provenance = dict(provenance)
            old_sources = dict(old_provenance.pop('engine_sources'))
            new_sources = dict(new_provenance.pop('engine_sources'))
            old_sources.pop('redirects')
            new_sources.pop('redirects')
            assert old_provenance == new_provenance and old_sources == new_sources
            assembly = saved['model_fields']['assembly']
            nodes, linkers, degrees = assembly_coordinates(assembly)
            lattice = CentralLattice.from_assemblies(reference, assembly).enumerate(nodes)
            envelope = FilledEnvelope(nodes, lattice.distance, alpha=1.05)
            primary = analyze(lattice, nodes, linkers, degrees, envelope=envelope)
            dest = out/path.parent.name
            dest.mkdir()
            for kind in ('node', 'linker', 'observed_node', 'observed_linker'):
                rows = primary.pop(kind+'_rows')
                if rows:
                    with (dest/(kind+'_filling.csv')).open('w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
            write_json(dest/'primary.json', primary)
            sensitivity = []
            for boundary in (.2, .35, .5):
                for tolerance in (.5, 1., 2.):
                    r = analyze(lattice, nodes, linkers, degrees, boundary=boundary,
                                tolerance=tolerance, detail=False, envelope=envelope)
                    sensitivity.append({'anchor': 0, 'alpha': 1.05, 'boundary': boundary,
                                        'tolerance': tolerance, 'result': r})
            for alpha in (.85, 1.25):
                r = analyze(lattice, nodes, linkers, degrees, alpha=alpha, detail=False)
                sensitivity.append({'anchor': 0, 'alpha': alpha, 'boundary': .35,
                                    'tolerance': .5, 'result': r})
            for anchor in (1, 2):
                alternative = CentralLattice.from_assemblies(reference, assembly, anchor).enumerate(nodes)
                r = analyze(alternative, nodes, linkers, degrees, detail=False, envelope=envelope)
                sensitivity.append({'anchor': anchor, 'alpha': 1.05, 'boundary': .35,
                                    'tolerance': .5, 'result': r})
            write_json(dest/'sensitivity.json', sensitivity)
            record.update(status='completed', N=len(assembly.entities),
                          tau=saved['simulation_fields']['tau'], primary=primary)
        except Exception:
            record.update(status='measurement_unresolved', error=traceback.format_exc())
        assert sha(path) == source_sha, 'Read-only measurement modified structure'
        records.append(record)
        write_json(out/'summary.json', {'snapshots': records, 'read_only': True})
    if not records:
        write_json(out/'summary.json', {'snapshots': [], 'status': 'no_readable_checkpoint'})
        raise SystemExit(2)
    resolved = sum(r['status'] == 'completed' for r in records)
    print(json.dumps({'trace': str(args.trace), 'snapshots': len(records), 'resolved': resolved}))
    if resolved != len(records):
        raise SystemExit(3)


if __name__ == '__main__':
    main()
