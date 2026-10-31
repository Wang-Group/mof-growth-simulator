"""Explicitly bounded single-trajectory interface; run only on 5090."""
import argparse
import json
from pathlib import Path

from runtime import configure, make_model, sha
from trajectory import Simulation, run_bounded


def main():
    ap = argparse.ArgumentParser()
    for name in ('root', 'bundle', 'preflight', 'acetate-source', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--mode', choices=['fixed', 'finite'], required=True)
    ap.add_argument('--fragment-policy', choices=['retain_oligomers', 'redissolve'], required=True)
    ap.add_argument('--delta', type=float, required=True)
    ap.add_argument('--rho', type=float, required=True)
    ap.add_argument('--s-close', type=float, default=1.)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--free-zr', type=int, default=1000)
    ap.add_argument('--free-bdc', type=int, default=6000)
    ap.add_argument('--target', type=int, required=True)
    ap.add_argument('--max-events', type=int, required=True)
    ap.add_argument('--wall-seconds', type=float, required=True)
    ap.add_argument('--model-time-limit', type=float)
    args = ap.parse_args()
    engine, provenance = configure(args.root, args.bundle, args.preflight)
    provenance['integration_sources'] = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    m = make_model(engine, args.root, args.acetate_source, seed=args.seed,
                   delta=args.delta, mode=args.mode, fragment_policy=args.fragment_policy,
                   rho=args.rho, s_close=args.s_close, free=(args.free_zr, args.free_bdc))
    provenance['acetate'] = m.library.provenance
    provenance['seed_sha256'] = sha(args.root/'framework_inputs/UiO66_N76_reviewed.pkl')
    sim = Simulation(m, args.seed ^ 0xFD7)
    result = run_bounded(sim, args.output, provenance, target=args.target,
                         max_events=args.max_events, wall_seconds=args.wall_seconds,
                         model_time_limit=args.model_time_limit)
    print(json.dumps({k: result[k] for k in ('status', 'events', 'final_N', 'maximum_N', 'tau', 'wall_seconds')}))


if __name__ == '__main__':
    main()
