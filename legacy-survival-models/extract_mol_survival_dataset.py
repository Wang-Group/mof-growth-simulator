import argparse
import csv
import pickle
import re
from pathlib import Path

import numpy as np


KEY_PATTERN = re.compile(
    r"Zr_(?P<zr>[-+0-9.eE]+)"
    r"_FA_(?P<fa>[-+0-9.eE]+)"
    r"_L_(?P<linker>[-+0-9.eE]+)"
    r"_Ratio_(?P<ratio>[-+0-9.eE]+)"
    r"_Step_(?P<step>[-+0-9.eE]+)"
    r"_SC_(?P<sc>[-+0-9.eE]+)"
    r"_KC_(?P<kc>[-+0-9.eE]+)"
)


def parse_legacy_number(text):
    cleaned = str(text).strip()
    cleaned = re.sub(r"([eE][+-]?\d+)\.0$", r"\1", cleaned)
    return float(cleaned)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MOL_batch0.pkl into a flat survival-ready CSV."
    )
    parser.add_argument("--batch-pkl", required=True, help="Path to MOL_batch0.pkl.")
    parser.add_argument(
        "--target-entities",
        type=float,
        default=20.0,
        help="Entity threshold used to define the event. Default: 20.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=0.24 / 1000.0,
        help="Scale factor converting the stored raw time into seconds. Default matches ml_v1.ipynb.",
    )
    parser.add_argument("--output-csv", required=True, help="Output flat CSV path.")
    return parser.parse_args()


def parse_condition_key(condition_key):
    match = KEY_PATTERN.search(condition_key)
    if not match:
        raise ValueError(f"Could not parse condition key: {condition_key}")
    parsed = {name: parse_legacy_number(value) for name, value in match.groupdict().items()}
    parsed["condition_id"] = condition_key
    return parsed


def main():
    args = parse_args()

    batch_path = Path(args.batch_pkl)
    with open(batch_path, "rb") as handle:
        batch = pickle.load(handle)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "system",
        "source_file",
        "condition_id",
        "replicate_index",
        "zr_mM",
        "fa_mM",
        "linker_mM",
        "ratio",
        "step",
        "sc",
        "kc",
        "target_entities",
        "raw_time",
        "time_seconds",
        "observed_entities",
        "event",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for condition_key, replicate_values in batch.items():
            parsed = parse_condition_key(condition_key)
            for replicate_index, entry in enumerate(replicate_values):
                arr = np.asarray(entry).reshape(-1)
                if arr.size < 2:
                    continue
                raw_time = float(arr[0])
                observed_entities = float(arr[1])
                event = int(observed_entities >= args.target_entities)

                writer.writerow(
                    {
                        "system": "BTB-MOL_legacy_ml",
                        "source_file": str(batch_path),
                        "condition_id": parsed["condition_id"],
                        "replicate_index": replicate_index,
                        "zr_mM": parsed["zr"],
                        "fa_mM": parsed["fa"],
                        "linker_mM": parsed["linker"],
                        "ratio": parsed["ratio"],
                        "step": parsed["step"],
                        "sc": parsed["sc"],
                        "kc": parsed["kc"],
                        "target_entities": args.target_entities,
                        "raw_time": raw_time,
                        "time_seconds": raw_time * args.time_scale,
                        "observed_entities": observed_entities,
                        "event": event,
                    }
                )


if __name__ == "__main__":
    main()
