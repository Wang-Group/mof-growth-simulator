"""Freeze the tested release and generate the approved 36 x 32 grid on 5090."""
import itertools
import json
from pathlib import Path
import shutil
import socket
from datetime import datetime, timezone
from campaign_io import sha, write_json

BASE = Path(__file__).resolve().parent
BUNDLE = Path('/home/wave/MOL_growth_recalculation_20260830')
INTEGRATION = BUNDLE / 'defect_integration_20260912_v1'
CENTRAL = BUNDLE / 'central_filling_analysis_20260912_v1'


def main():
    assert socket.gethostname() == 'xmhy-G3261'
    assert not (BASE / 'manifest.json').exists(), 'Never replace a frozen campaign'
    release = json.loads((INTEGRATION / 'VALIDATED_RELEASE.json').read_text())
    copies = []
    for original, meta in release['sources'].items():
        original = Path(original)
        if original.is_relative_to(INTEGRATION):
            destination = BASE / 'frozen/integration' / original.relative_to(INTEGRATION)
        elif original.is_relative_to(BUNDLE):
            destination = BASE / 'frozen/bundle' / original.relative_to(BUNDLE)
        else:
            assert original.name == 'uio66__Zr6_AA.mol'
            destination = BASE / 'frozen/acetate.mol'
        source = INTEGRATION / meta['frozen_copy']
        assert sha(source) == meta['sha256']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copies.append({'original': str(original), 'frozen': str(destination.relative_to(BASE)),
                       'sha256': sha(destination)})
    for p in (INTEGRATION / 'validated_release/integration_source').glob('*.py'):
        dest = BASE / 'frozen/integration' / p.name
        if dest.exists():
            assert sha(dest) == sha(p)
        else:
            shutil.copy2(p, dest)
    (BASE / 'frozen/analysis').mkdir(parents=True)
    for name in ('central_filling.py', 'test_central_filling.py'):
        shutil.copy2(CENTRAL / name, BASE / 'frozen/analysis' / name)
    shutil.copy2(INTEGRATION / 'VALIDATED_RELEASE.json', BASE / 'frozen/VALIDATED_RELEASE.json')
    traces = []
    # Replicate blocks pair the initial random streams across all conditions.
    # Every condition has 32 independent replicate seeds, not 1152 independent
    # chemistry conditions. Launch order alternates low/high penalties.
    for rep, delta, rho, mode in itertools.product(range(32), (0, 10, 2, 8, 4, 6), (.1, 1., 3.), ('finite', 'fixed')):
        condition = f'd{delta:02d}_rho{str(rho).replace(".", "p")}_{mode}'
        traces.append({'id': f'{condition}_r{rep:02d}', 'condition': condition,
                       'replicate': rep, 'seed': 2026091300 + rep,
                       'delta': delta, 'rho0': rho, 'mode': mode,
                       'fragment_policy': 'retain_oligomers'})
    assert len(traces) == 1152 and len({t['id'] for t in traces}) == 1152
    files = {str(p.relative_to(BASE)): sha(p) for p in sorted(BASE.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts}
    manifest = {
        'schema': 'acetate-50000steps-production-1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'campaign_id': BASE.name, 'host': socket.gethostname(), 'files': files,
        'source_copy_map': copies, 'target_N': None, 'target_events': 50000, 'target_kind': 'event_steps', 'N_start': 76,
        'replaces_campaign': 'defect_N5000_production_20260913_v2',
        'restart_policy': 'fresh reviewed N76 seed with same condition and replicate RNG seeds; old outcomes preserved separately',
        'N_definition': 'number of Zr6 entities plus BDC molecules; no caps/proxies',
        'initial_free': {'Zr6': 1000, 'BDC': 6000}, 'initial_free_atomic_Zr_BDC': 1.0,
        's_close': 1.0, 'eta_solvent_prior': .5,
        'delta_definition': 'U_D/(RT); extra F and V energies zero; no extra shared TS penalty',
        'rho0_definition': 'fixed per-bond removal coefficient in initial-total-external-addition time units',
        'clock': 'tau=k_external_total_initial*t; fixed throughout each trajectory',
        'milestones_N': [250, 500, 1000, 2000, 5000],
        'milestones_steps': [1000, 5000, 10000, 25000, 50000],
        'curve_event_interval': 100, 'progress_wall_interval_seconds': 30,
        'checkpoint_wall_interval_seconds': 1800,
        'wall_seconds': 3600, 'common_model_time_cap': None,
        'workers': {'initial': 8, 'maximum': 64, 'ramp_after_seconds': 120,
                    'threads_per_process': 1, 'memory_reserve_GiB': 24,
                    'minimum_worker_memory_GiB': 1, 'observed_worker_peak_safety_factor': 3},
        'analysis': {'python': str(CENTRAL / '.analysis_venv/bin/python'), 'workers': 2,
                     'alpha': 1.05, 'boundary_dNN': .35, 'tolerance_A': .5,
                     'deep_boundary_dNN': .5, 'sensitivity': 'boundary .2/.35/.5; tolerance .5/1/2; alpha .85/1.05/1.25; 3 anchors'},
        'condition_count': 36, 'replicates_per_condition': 32, 'trace_count': len(traces),
        'hypothesis': 'DEF-INT-01: unfavorable singly attached intermediate can inhibit vacancy repair during depletion',
        'null': 'DEF-NULL-01: no persistent interior vacancy enrichment versus delta=0 at comparable size/time',
        'interpretation': [
            'Seeded growth and defect survival, not unseeded nucleation.',
            'No monotonic defect trend is imposed: intermediate penalty hinders both creation and repair.',
            'Fixed acid/solvent reservoirs; finite Zr6/BDC pools are molecule-count inventories, not a fitted concentration recipe.',
            'Detached oligomers leave the tracked structure but remain counted and unavailable; released monomers recycle.',
            'Complete at exactly 50000 attempted events, including rejections and cap exchange; there is no N-based termination.',
            'One-hour wall cap is censoring before the step target, not physical growth termination; errors are separate invalid runs.',
            'Equal event counts are not equal model times: report tau, final N and maximum N alongside defect counts.',
            'No extrapolation of a censored curve to unobserved common model time.',
            'Surface/boundary sensitivity and unmatched positions must accompany any interior-defect claim.',
            'Only framework-only XYZ is suitable as geometric input for later framework PXRD; solvent proxy is not a physical scatterer.'
        ], 'traces': traces}
    write_json(BASE / 'manifest.json', manifest)
    print(json.dumps({'manifest': str(BASE / 'manifest.json'), 'sha256': sha(BASE / 'manifest.json'),
                      'traces': len(traces), 'files': len(files)}))


if __name__ == '__main__':
    main()
