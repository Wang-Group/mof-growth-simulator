import argparse
import json
import random
import re
from pathlib import Path

import numpy as np


DEFAULTS = {
    "ZR6_PERCENTAGE": 0.6,
    "Zr_conc": 22.9154705859894,
    "entropy_correction_coefficient": 0.789387907185137,
    "equilibrium_constant_coefficient": 1.32975172557788,
    "H2O_DMF_RATIO": 3e-10,
    "Capping_agent_conc": 1473.06341756944,
    "Linker_conc": 69.1596872253079,
    "Total_steps": 200000,
    "current_folder": None,
    "BUMPING_THRESHOLD": 2,
    "pkl_path": None,
    "max_entities": 120,
    "output_inter": 10,
    "last_saved": -1,
    "MAX_SIM_TIME_SECONDS": None,
    "DISSOLUTION_UPDATE_INTERVAL_STEPS": None,
    "EXCHANGE_RXN_TIME_SECONDS": 0.1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run UiO66_growth_main_conc.py from a seed PKL with explicit parameters."
    )
    parser.add_argument("--seed-pkl", required=True, help="Input assembly pickle to continue from.")
    parser.add_argument("--output-dir", required=True, help="Output directory for the growth run.")
    parser.add_argument("--rng-seed", type=int, default=0, help="Seed for Python and NumPy RNGs.")
    parser.add_argument("--max-entities", type=int, default=DEFAULTS["max_entities"])
    parser.add_argument("--output-inter", type=int, default=DEFAULTS["output_inter"])
    parser.add_argument("--total-steps", type=int, default=DEFAULTS["Total_steps"])
    parser.add_argument("--max-sim-time-seconds", type=float, default=DEFAULTS["MAX_SIM_TIME_SECONDS"])
    parser.add_argument(
        "--exchange-rxn-time-seconds",
        type=float,
        default=DEFAULTS["EXCHANGE_RXN_TIME_SECONDS"],
    )
    parser.add_argument(
        "--dissolution-update-interval-steps",
        type=int,
        default=DEFAULTS["DISSOLUTION_UPDATE_INTERVAL_STEPS"],
    )
    parser.add_argument("--zr6-percentage", type=float, default=DEFAULTS["ZR6_PERCENTAGE"])
    parser.add_argument("--zr-conc", type=float, default=DEFAULTS["Zr_conc"])
    parser.add_argument(
        "--entropy-correction-coefficient",
        type=float,
        default=DEFAULTS["entropy_correction_coefficient"],
    )
    parser.add_argument(
        "--equilibrium-constant-coefficient",
        type=float,
        default=DEFAULTS["equilibrium_constant_coefficient"],
    )
    parser.add_argument("--h2o-dmf-ratio", type=float, default=DEFAULTS["H2O_DMF_RATIO"])
    parser.add_argument("--capping-agent-conc", type=float, default=DEFAULTS["Capping_agent_conc"])
    parser.add_argument("--linker-conc", type=float, default=DEFAULTS["Linker_conc"])
    parser.add_argument("--bumping-threshold", type=float, default=DEFAULTS["BUMPING_THRESHOLD"])
    parser.add_argument(
        "--template",
        default="UiO66_growth_main_conc.py",
        help="Path to the growth template script.",
    )
    return parser.parse_args()


def replace_assignment(source, key, value):
    pattern = rf"^{re.escape(key)}\s*=.*$"
    replacement = f"{key} = {repr(value)}"
    updated_source, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Failed to set {key} in template")
    return updated_source


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    seed_pkl = (script_dir / args.seed_pkl).resolve() if not Path(args.seed_pkl).is_absolute() else Path(args.seed_pkl)
    output_dir = (script_dir / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    template_path = (script_dir / args.template).resolve() if not Path(args.template).is_absolute() else Path(args.template)

    config = {
        "ZR6_PERCENTAGE": args.zr6_percentage,
        "Zr_conc": args.zr_conc,
        "entropy_correction_coefficient": args.entropy_correction_coefficient,
        "equilibrium_constant_coefficient": args.equilibrium_constant_coefficient,
        "H2O_DMF_RATIO": args.h2o_dmf_ratio,
        "Capping_agent_conc": args.capping_agent_conc,
        "Linker_conc": args.linker_conc,
        "Total_steps": args.total_steps,
        "current_folder": output_dir.as_posix(),
        "BUMPING_THRESHOLD": args.bumping_threshold,
        "pkl_path": seed_pkl.as_posix(),
        "max_entities": args.max_entities,
        "output_inter": args.output_inter,
        "last_saved": -1,
        "MAX_SIM_TIME_SECONDS": args.max_sim_time_seconds,
        "DISSOLUTION_UPDATE_INTERVAL_STEPS": args.dissolution_update_interval_steps,
        "EXCHANGE_RXN_TIME_SECONDS": args.exchange_rxn_time_seconds,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.rng_seed)
    np.random.seed(args.rng_seed)

    source = template_path.read_text(encoding="utf-8")
    source = source.replace(
        "from IPython.display import clear_output",
        "try:\n    from IPython.display import clear_output\nexcept ImportError:\n    def clear_output(*args, **kwargs):\n        return None",
        1,
    )
    for key, value in config.items():
        source = replace_assignment(source, key, value)

    print("Running seeded growth with config:")
    print(json.dumps(config, indent=2))
    print(f"rng_seed = {args.rng_seed}")

    exec_globals = {
        "__name__": "__main__",
        "__file__": str(template_path),
    }
    exec(compile(source, str(template_path), "exec"), exec_globals)


if __name__ == "__main__":
    main()
