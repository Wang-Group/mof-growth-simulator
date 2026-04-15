import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from distorted_ligand_model import (
    PREBOUND_MODEL_CLUSTER_ONE_TO_ONE,
    compute_prebound_chemistry_state,
)


DEFAULTS = {
    "ZR6_PERCENTAGE": 1.0,
    "Zr_conc": 32.0,
    "entropy_correction_coefficient": 0.789387907185137,
    "equilibrium_constant_coefficient": 1.32975172557788,
    "H2O_DMF_RATIO": 0.0,
    "Capping_agent_conc": 300.0,
    "Linker_conc": 4.0,
    "Total_steps": 100000,
    "BUMPING_THRESHOLD": 2.0,
    "max_entities": 60,
    "output_inter": 100000,
    "EXCHANGE_RXN_TIME_SECONDS": 0.1,
    "DISSOLUTION_UPDATE_INTERVAL_STEPS": 1000000,
    "DISTORTED_LINKER_ENABLED": False,
    "DISTORTED_CHEMISTRY_MODEL": PREBOUND_MODEL_CLUSTER_ONE_TO_ONE,
    "DISTORTED_LIGAND_ASSOCIATION_CONSTANT": None,
    "DISTORTED_SITE_EQUILIBRIUM_CONSTANT": None,
    "DISTORTED_SECOND_STEP_EQUIVALENTS": 0.0,
    "DISTORTED_NUM_SITES_ON_CLUSTER": 12,
    "DISTORTED_NUM_SITES_ON_LINKER": 2,
    "Index": 0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a UiO-66 case from the normal code baseline, optionally enabling "
            "a prebound Zr-BDC branch derived from the existing KC chemistry."
        )
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
    parser.add_argument("--total-steps", type=int, default=DEFAULTS["Total_steps"])
    parser.add_argument("--bumping-threshold", type=float, default=DEFAULTS["BUMPING_THRESHOLD"])
    parser.add_argument("--max-entities", type=int, default=DEFAULTS["max_entities"])
    parser.add_argument("--output-inter", type=int, default=DEFAULTS["output_inter"])
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
    parser.add_argument(
        "--enable-distorted-linker",
        action="store_true",
        help="Enable the distorted-linker branch derived from KC.",
    )
    parser.add_argument(
        "--distorted-chemistry-model",
        default=DEFAULTS["DISTORTED_CHEMISTRY_MODEL"],
        help=(
            "Chemistry model for the prebound branch. "
            "Use 'cluster_one_to_one' for the old coarse model or "
            "'multisite_first_binding_only' for the 12-site/2-site model."
        ),
    )
    parser.add_argument(
        "--distorted-ligand-association-constant",
        type=float,
        default=DEFAULTS["DISTORTED_LIGAND_ASSOCIATION_CONSTANT"],
        help="Optional direct override for the effective distorted-linker association constant.",
    )
    parser.add_argument(
        "--distorted-site-equilibrium-constant",
        type=float,
        default=DEFAULTS["DISTORTED_SITE_EQUILIBRIUM_CONSTANT"],
        help="Optional direct override for the multisite site-level exchange constant.",
    )
    parser.add_argument(
        "--distorted-second-step-equivalents",
        type=float,
        default=DEFAULTS["DISTORTED_SECOND_STEP_EQUIVALENTS"],
        help=(
            "Optional extra irreversible sink for the older two-step model. "
            "Leave at 0 to keep only the 1:1 prebound Zr-BDC species."
        ),
    )
    parser.add_argument(
        "--distorted-num-sites-on-cluster",
        type=int,
        default=DEFAULTS["DISTORTED_NUM_SITES_ON_CLUSTER"],
    )
    parser.add_argument(
        "--distorted-num-sites-on-linker",
        type=int,
        default=DEFAULTS["DISTORTED_NUM_SITES_ON_LINKER"],
    )
    parser.add_argument("--index", type=int, default=DEFAULTS["Index"])
    parser.add_argument(
        "--output-root",
        default="output/zr_ligand_cases",
        help="Parent directory that will receive the timestamped run folder.",
    )
    parser.add_argument(
        "--template",
        default="UiO66_growth_main_20250811.py",
        help="Path to the growth template script.",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="Optional explicit folder name. By default it follows the chemistry naming scheme.",
    )
    return parser.parse_args()


def replace_assignment(source, key, value):
    pattern = rf"^{re.escape(key)}\s*=.*$"
    replacement = f"{key} = {repr(value)}"
    updated_source, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Failed to set {key} in template")
    return updated_source


def tokenise_float(value):
    if value is None:
        return "auto"
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def build_folder_name(args, timestamp):
    step_mag = int(len(str(int(args.total_steps))) - 1) if args.total_steps > 0 else 0
    distorted_token = "on" if args.enable_distorted_linker else "off"
    model_token = str(args.distorted_chemistry_model).replace("-", "_")
    return (
        f"Zr_{args.zr_conc}_FA_{args.capping_agent_conc}_L_{args.linker_conc}"
        f"_Ratio_{args.h2o_dmf_ratio}_Step_1e{step_mag}_Index_{args.index}"
        f"_SC_{args.entropy_correction_coefficient}_KC_{args.equilibrium_constant_coefficient}"
        f"_Nmax_{args.max_entities}_DL_{distorted_token}_PM_{model_token}"
        f"_AK_{tokenise_float(args.distorted_ligand_association_constant)}"
        f"_SK_{tokenise_float(args.distorted_site_equilibrium_constant)}"
        f"_S2_{tokenise_float(args.distorted_second_step_equivalents)}"
        f"_{timestamp}"
    )


def resolve_output_dir(output_root, folder_name):
    path = Path(output_root)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    run_dir = path / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    folder_name = args.basename or build_folder_name(args, timestamp)
    output_dir = resolve_output_dir(args.output_root, folder_name)
    template_path = Path(args.template)
    if not template_path.is_absolute():
        template_path = (Path(__file__).resolve().parent / template_path).resolve()

    chemistry_preview = (
        compute_prebound_chemistry_state(
            zr_conc=args.zr_conc,
            linker_conc=args.linker_conc,
            equilibrium_constant_coefficient=args.equilibrium_constant_coefficient,
            h2o_dmf_ratio=args.h2o_dmf_ratio,
            capping_agent_conc=args.capping_agent_conc,
            zr6_percentage=args.zr6_percentage,
            model_name=args.distorted_chemistry_model,
            association_constant_override=args.distorted_ligand_association_constant,
            site_equilibrium_constant_override=args.distorted_site_equilibrium_constant,
            dimethylamine_conc=0.0,
            second_step_equivalents=args.distorted_second_step_equivalents,
            num_sites_on_cluster=args.distorted_num_sites_on_cluster,
            num_sites_on_linker=args.distorted_num_sites_on_linker,
        )
        if args.enable_distorted_linker
        else None
    )

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
        "max_entities": args.max_entities,
        "output_inter": args.output_inter,
        "EXCHANGE_RXN_TIME_SECONDS": args.exchange_rxn_time_seconds,
        "DISSOLUTION_UPDATE_INTERVAL_STEPS": args.dissolution_update_interval_steps,
        "DISTORTED_LINKER_ENABLED": args.enable_distorted_linker,
        "DISTORTED_CHEMISTRY_MODEL": args.distorted_chemistry_model,
        "DISTORTED_LIGAND_ASSOCIATION_CONSTANT": args.distorted_ligand_association_constant,
        "DISTORTED_SITE_EQUILIBRIUM_CONSTANT": args.distorted_site_equilibrium_constant,
        "DISTORTED_SECOND_STEP_EQUIVALENTS": args.distorted_second_step_equivalents,
        "DISTORTED_NUM_SITES_ON_CLUSTER": args.distorted_num_sites_on_cluster,
        "DISTORTED_NUM_SITES_ON_LINKER": args.distorted_num_sites_on_linker,
    }

    launcher_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "template_path": template_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "config": config,
        "chemistry_preview": chemistry_preview,
    }
    (output_dir / "launcher_config.json").write_text(
        json.dumps(launcher_payload, indent=2),
        encoding="utf-8",
    )

    source = template_path.read_text(encoding="utf-8")
    for key, value in config.items():
        source = replace_assignment(source, key, value)

    print("Running Zr-Ligand case with config:")
    print(json.dumps(launcher_payload, indent=2))

    exec_globals = {
        "__name__": "__main__",
        "__file__": str(template_path),
    }
    exec(compile(source, str(template_path), "exec"), exec_globals)


if __name__ == "__main__":
    main()
