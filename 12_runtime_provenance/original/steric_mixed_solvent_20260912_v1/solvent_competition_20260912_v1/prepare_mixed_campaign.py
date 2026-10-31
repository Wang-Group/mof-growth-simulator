"""Generate new, non-overwriting mixed-solvent design manifests (no simulations)."""
from pathlib import Path
import csv
import hashlib
import json


def main():
    here = Path(__file__).resolve().parent
    source = here.parent / 'proposed_production_manifest.csv'
    with source.open(newline='') as f:
        base = list(csv.DictReader(f))
    grid = []
    for row in base:
        for eta in (0.1, 0.5, 0.9):
            r = dict(row)
            r['run_id'] = row['run_id'] + f'_eta{eta}'
            r['eta_S'] = eta
            r['status'] = 'AUTHORIZED_MIXED_SOLVENT_TEST_THEN_GRID'
            r['model_time_limit'] = 100.0 if row['assay'] == 'nucleation' else 20.0
            r['termination_model'] = 'explicit_M_S_exchange_cap_free_ready'
            grid.append(r)
    pilot = [r for r in grid if float(r['rho_off']) == 1 and float(r['q_atomic_Zr_BDC']) == 1
             and int(r['replicate']) < 4]
    assert len(grid) == 27648 and len(pilot) == 288
    for name, rows in [('grid_manifest.csv', grid), ('pilot_manifest.csv', pilot)]:
        with (here / name).open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plan = {
        'author_authorization': 'please go ahead with the test and grid run',
        'source_manifest_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'identities': sorted({r['identity'] for r in grid}),
        'rho_off': [0.1, 1, 3, 10], 'q_atomic_Zr_BDC': [0.3, 1, 3],
        'eta_S_uncrowded': [0.1, 0.5, 0.9], 's_close': 1,
        'conditions': 432, 'assays': ['nucleation N1->30', 'seeded growth N76->126'],
        'replicates_per_assay': 32, 'grid_records': len(grid), 'pilot_records': len(pilot),
        'model_time_windows': {'nucleation': 100, 'growth': 20},
        'wall_caps_seconds': {'nucleation': 120, 'growth': 600},
        'event_caps': {'nucleation': 1000000, 'growth': 5000000},
        'pilot_workers': 16, 'maximum_grid_workers': 96,
        'replicate_geometry_exports': 'replicate 0; all invalid dynamic records additionally retained',
        'large_raw_artifacts_remote': True,
        'analysis': 'first-passage probabilities with censoring and time distributions; conditional growth speed labelled; invalid geometry separate',
        'pilot_and_grid_overlap': 'pilot replicates are deterministic paired repeats; do not pool pilot and grid as independent data',
        'random_seed_pairing': 'seed label common across identities and eta at fixed rho,q,assay,replicate',
        'not_measured_occupancies': True,
        'pre_result_predictions': [
            'Steric discrimination may decrease as solvent preference increases; this is a hypothesis, not a required result',
            'Uncrowded chemical framework coefficients remain identical at matched rho,q,s_close',
            'Crowded-site solvent occupancy may exceed its uncrowded preference',
            'Neither null identity differences nor remaining geometry failures will be hidden']}
    with (here / 'CAMPAIGN_DESIGN.json').open('x') as f:
        json.dump(plan, f, indent=2)
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
