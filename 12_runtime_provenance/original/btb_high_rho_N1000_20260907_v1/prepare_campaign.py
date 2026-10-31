"""Prepare only the approved 4 x 8 pilot; no simulation is launched here."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMP = HERE.parent
NAME = HERE.name
REMOTE_ROOT = '/home/wave/MOL_growth_recalculation_20260830'
REMOTE = f'{REMOTE_ROOT}/{NAME}'
SOURCE = CAMP / 'student_handoff_20260906_v4_light/04_simulation_code_and_inputs/verified_5090_snapshot'
SEED = 'reviewed_structures/btb_q10_off10_sc2_seed1708319965_n30.pkl'
FILES = [
    'code/proposed/run_seeded_manifest.py',
    'code/proposed/seeded_molecular_runner.py',
    'code/proposed/common_molecular_runner.py',
    'code/proposed/common_kmc_selector.py',
    'code/proposed/btb_postgrowth_structure.py',
    'code/proposed/test_common_kmc_selector.py',
    'code/candidate_corrected/MOL_Assembly_Large_Correction_20250811_candidate.py',
    *[f'models/workstation_source_20260831/mol/{n}.mol' for n in ('Zr6', 'Zr12', 'BTBa', 'BTBb')],
    SEED,
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_once(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(value)

def main():
    expected = {name: sha(SOURCE / name) for name in FILES}
    assert expected[SEED] == 'e98bb24e620e05213755b49755dc2723ff7309d718870f60a96bc4fc8a0281ee'
    assert expected[FILES[1]] == 'b5354813246ae10e5013af58422d980fe092323ebab1e7926263a6d8bccf527b'
    template_path = CAMP / 'btb_N10000_pxrd_supplement_20260907_v1/scripts/campaign_inputs/manifest.csv'
    with template_path.open(encoding='utf-8-sig', newline='') as f:
        template = next(csv.DictReader(f))
    rows = []
    # Interleave conditions so both worker waves cover the full range.
    for rep in range(8):
        for rho in (10, 30, 50, 100):
            condition = f'btb_high_rho_N1000_sc1_q1_rho{rho}'
            run_id = f'{condition}_pilot20260907_rep{rep:02d}'
            random_seed = int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:4], 'big')
            row = dict(template)
            row.update(run_id=run_id, condition_id=condition,
                       design_block=NAME, off_weight=str(rho),
                       seed_pickle=f'{REMOTE}/snapshot/{SEED}',
                       replicate=str(rep), continuation_replicate=str(rep),
                       random_seed=str(random_seed), target_entities='1000',
                       added_entity_target='970', event_cap='100000000',
                       wall_time_cap_s='14400', structure_metric_milestones='250;500;1000',
                       mol2_milestones='250;500;1000', stage='high_rho_pilot_8_replicates')
            rows.append(row)
    assert len({r['random_seed'] for r in rows}) == len(rows) == 32
    assert all(float(r['SC']) == 1 and float(r['node_fraction']) == 1/7 for r in rows)
    with (HERE / 'manifest.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    old = json.loads((CAMP / 'btb_large_pxrd_convergence_20260905/remote_N1000_results/btb_large_pxrd_N1000_sc1_q1_rho10_rep00.json').read_text(encoding='utf-8'))
    design = dict(name=NAME, remote_root=REMOTE_ROOT, remote_output=REMOTE,
                  workers=16, rho_off=[10,30,50,100], replicates_per_condition=8,
                  target_entities=1000, initial_entities=30, SC=1,
                  atomic_metal_to_BTB_ratio=1, node_fraction=1/7,
                  external_activity=1, event_cap=100000000, wall_time_cap_s=14400,
                  milestones=[250,500,1000], expected_source_sha256=expected,
                  seed_relative_path=SEED, manifest_sha256=sha(HERE / 'manifest.csv'),
                  baseline_smoke=dict(random_seed=old['random_seed'],
                                      rho_off=10, target_entities=1000,
                                      stop_event=old['first_passage_event']['80'],
                                      expected_first_passage_time_80=old['first_passage_model_time']['80']),
                  scope='32 pilot trajectories only. No automatic expansion or cap increase.',
                  kinetics='Unmodified reviewed runner and selector; full 3D; all bonds removable equally.',
                  runtime='Same /usr/bin/python3 runtime and existing dependency adapters as baseline.',
                  limitations=['One fixed N=30 seed geometry; replicate uncertainty is conditional on this seed.',
                               'Archived transitive layer-domain counts are not literal physical layer counts.',
                               'NMR clock field retained for compatibility, not used to claim productive-growth seconds.',
                               'N=1000 means total entities (clusters plus BTB), not 1000 nodes or unit cells.'])
    write_once(HERE / 'design.json', json.dumps(design, indent=2) + '\n')
    print(json.dumps(dict(rows=len(rows), workers=16, manifest_sha256=design['manifest_sha256'],
                          smoke=design['baseline_smoke']), indent=2))

if __name__ == '__main__':
    main()
