import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from multisite_linker_exchange_model import compute_multisite_exchange_state


DEFAULTS = {
    "ZR6_PERCENTAGE": 1.0,
    "Zr_conc": 32.0,
    "equilibrium_constant_coefficient": 1.32975172557788,
    "SITE_EQUILIBRIUM_CONSTANT": None,
    "H2O_DMF_RATIO": 0.0,
    "Capping_agent_conc": 300.0,
    "Linker_conc": 4.0,
    "dimethylamine_conc": 0.0,
    "num_sites_on_cluster": 12,
    "num_sites_on_linker": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a standalone multisite Zr-AA/linker exchange equilibrium with "
            "12 cluster sites and 2 linker sites, without any follow-up reactions."
        )
    )
    parser.add_argument("--zr6-percentage", type=float, default=DEFAULTS["ZR6_PERCENTAGE"])
    parser.add_argument("--zr-conc", type=float, default=DEFAULTS["Zr_conc"])
    parser.add_argument(
        "--equilibrium-constant-coefficient",
        type=float,
        default=DEFAULTS["equilibrium_constant_coefficient"],
        help="KC-style multiplier used to derive the site-level K_eff.",
    )
    parser.add_argument(
        "--site-equilibrium-constant",
        type=float,
        default=DEFAULTS["SITE_EQUILIBRIUM_CONSTANT"],
        help="Optional direct override for the site-level exchange constant.",
    )
    parser.add_argument("--h2o-dmf-ratio", type=float, default=DEFAULTS["H2O_DMF_RATIO"])
    parser.add_argument("--capping-agent-conc", type=float, default=DEFAULTS["Capping_agent_conc"])
    parser.add_argument("--linker-conc", type=float, default=DEFAULTS["Linker_conc"])
    parser.add_argument("--dimethylamine-conc", type=float, default=DEFAULTS["dimethylamine_conc"])
    parser.add_argument("--num-sites-on-cluster", type=int, default=DEFAULTS["num_sites_on_cluster"])
    parser.add_argument("--num-sites-on-linker", type=int, default=DEFAULTS["num_sites_on_linker"])
    parser.add_argument(
        "--output-root",
        default="output/multisite_exchange_cases",
        help="Parent directory that will receive the timestamped output folder.",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="Optional explicit output folder name.",
    )
    return parser.parse_args()


def tokenise_float(value):
    if value is None:
        return "auto"
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def build_folder_name(args, timestamp):
    return (
        f"multisite_Zr_{tokenise_float(args.zr_conc)}"
        f"_FA_{tokenise_float(args.capping_agent_conc)}"
        f"_L_{tokenise_float(args.linker_conc)}"
        f"_KC_{tokenise_float(args.equilibrium_constant_coefficient)}"
        f"_SK_{tokenise_float(args.site_equilibrium_constant)}"
        f"_Zsites_{int(args.num_sites_on_cluster)}"
        f"_Lsites_{int(args.num_sites_on_linker)}"
        f"_{timestamp}"
    )


def resolve_output_dir(output_root, folder_name):
    path = Path(output_root)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    output_dir = path / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_distribution_csv(distribution_rows, output_path):
    fieldnames = [
        "bound_linkers_per_cluster",
        "cluster_fraction",
        "cluster_conc",
        "bound_linker_conc",
        "remaining_aa_sites_per_cluster",
        "remaining_aa_site_conc",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in distribution_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = args.basename or build_folder_name(args, timestamp)
    output_dir = resolve_output_dir(args.output_root, folder_name)

    state = compute_multisite_exchange_state(
        zr_conc=args.zr_conc,
        linker_conc=args.linker_conc,
        equilibrium_constant_coefficient=args.equilibrium_constant_coefficient,
        h2o_dmf_ratio=args.h2o_dmf_ratio,
        capping_agent_conc=args.capping_agent_conc,
        zr6_percentage=args.zr6_percentage,
        site_equilibrium_constant_override=args.site_equilibrium_constant,
        dimethylamine_conc=args.dimethylamine_conc,
        num_sites_on_cluster=args.num_sites_on_cluster,
        num_sites_on_linker=args.num_sites_on_linker,
    )

    summary_path = output_dir / "multisite_exchange_summary.json"
    distribution_path = output_dir / "cluster_occupancy_distribution.csv"

    summary_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    write_distribution_csv(state["cluster_occupancy_distribution"], distribution_path)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": output_dir.as_posix(),
        "summary_json": summary_path.as_posix(),
        "distribution_csv": distribution_path.as_posix(),
        "state": state,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
