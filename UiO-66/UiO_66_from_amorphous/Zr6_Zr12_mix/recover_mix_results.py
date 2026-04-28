import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "mixed_nuclei"
HISTORICAL_DIR = SCRIPT_DIR / "output" / "historical_reference"

SEED_BASENAME = "ratio_controlled_mixed_seed_160_seed612_frac055_fill6_regenerated"
RUN_BASENAME = "oneshot_regenerated_seed160_eq2_cap180_zr5000_pruned"


def run_python(script_name, *args):
    command = [sys.executable, script_name, *args]
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)


def copy_historical_reference_images():
    source_dir = (
        SCRIPT_DIR.parent
        / "initialize_structures1"
        / "output"
        / "mixed_nuclei"
        / "zr6_only_growth_runs"
        / "oneshot_seed160_eq2_cap180_zr5000_pruned"
    )
    copied = []
    for filename in [
        "oneshot_seed160_eq2_cap180_zr5000_pruned.summary_trace.svg",
        "oneshot_seed160_eq2_cap180_zr5000_pruned.summary_trace.png",
    ]:
        source = source_dir / filename
        if source.exists():
            destination = HISTORICAL_DIR / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(destination)
    return copied


def write_manifest():
    summary_json = OUTPUT_DIR / "zr6_only_growth_runs" / RUN_BASENAME / f"{RUN_BASENAME}.summary.json"
    trace_json = OUTPUT_DIR / "zr6_only_growth_runs" / RUN_BASENAME / f"{RUN_BASENAME}.summary_trace.json"
    trace_svg = OUTPUT_DIR / "zr6_only_growth_runs" / RUN_BASENAME / f"{RUN_BASENAME}.summary_trace.svg"
    seed_json = OUTPUT_DIR / f"{SEED_BASENAME}.json"
    seed_mol2 = OUTPUT_DIR / f"{SEED_BASENAME}.mol2"
    seed_pkl = OUTPUT_DIR / f"{SEED_BASENAME}.pkl"

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workspace_dir": SCRIPT_DIR.as_posix(),
        "seed_files": {
            "json": seed_json.as_posix(),
            "mol2": seed_mol2.as_posix(),
            "pkl": seed_pkl.as_posix(),
        },
        "run_outputs": {
            "summary_json": summary_json.as_posix(),
            "trace_json": trace_json.as_posix(),
            "trace_svg": trace_svg.as_posix(),
        },
        "historical_reference_dir": HISTORICAL_DIR.as_posix(),
    }

    if summary_json.exists():
        summary_payload = json.loads(summary_json.read_text(encoding="utf-8"))
        payload["summary_snapshot"] = {
            "seed_metadata": summary_payload.get("seed_metadata", {}),
            "run_count": len(summary_payload.get("runs", [])),
            "all_final_zr12_zero": all(
                row.get("end_counts", {}).get("Zr12_AA") == 0
                for row in summary_payload.get("runs", [])
            ),
        }

    manifest_path = SCRIPT_DIR / "output" / "recovery_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    run_python(
        "build_ratio_controlled_mixed_seed.py",
        "--rng-seed", "612",
        "--target-entities", "160",
        "--attempts-per-step", "1000",
        "--max-growth-steps", "30000",
        "--cluster-add-probability", "0.65",
        "--internal-link-probability", "0.20",
        "--target-zr12-fraction", "0.55",
        "--initial-internal-zr12-count", "5",
        "--seed-zr6-branches", "4",
        "--initial-zr6-branches-per-zr12", "1",
        "--inner-zr6-fill-count", "6",
        "--inner-zr6-radial-min", "0.08",
        "--inner-zr6-radial-max", "0.58",
        "--inner-zr6-target-radial", "0.32",
        "--max-zr12-coordination", "8",
        "--pick-preference", "sparse_outer",
        "--output-dir", "output/mixed_nuclei",
        "--basename", SEED_BASENAME,
    )

    run_python(
        "run_zr6_only_growth_case.py",
        "--seed-pkls", f"output/mixed_nuclei/{SEED_BASENAME}.pkl",
        "--replicates", "4",
        "--base-rng-seed", "2026040311",
        "--total-steps", "100000",
        "--max-entities-delta", "600",
        "--exchange-rxn-time-seconds", "0.1",
        "--zr-conc", "5000",
        "--linker-conc", "69.1596872253079",
        "--capping-agent-conc", "180",
        "--equilibrium-constant-coefficient", "2.0",
        "--entropy-correction-coefficient", "0.0",
        "--output-dir", "output/mixed_nuclei/zr6_only_growth_runs",
        "--basename", RUN_BASENAME,
    )

    run_python(
        "plot_zr6_only_growth_trace.py",
        "--summary-json", f"output/mixed_nuclei/zr6_only_growth_runs/{RUN_BASENAME}/{RUN_BASENAME}.summary.json",
        "--snapshot-every-steps", "1",
        "--mean-alignment", "time",
        "--mean-grid-points", "400",
        "--basename", f"{RUN_BASENAME}.summary_trace",
    )

    copied = copy_historical_reference_images()
    for path in copied:
        print(f"Copied historical reference: {path}")

    write_manifest()


if __name__ == "__main__":
    main()
