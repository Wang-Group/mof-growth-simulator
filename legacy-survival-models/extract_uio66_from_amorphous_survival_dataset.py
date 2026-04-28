import argparse
import csv
import re
from datetime import datetime
from pathlib import Path


PATH_PATTERN = re.compile(
    r"Zr_(?P<zr>[-+0-9.eE]+)"
    r"_FA_(?P<fa>[-+0-9.eE]+)"
    r"_L_(?P<linker>[-+0-9.eE]+)"
    r"_Ratio_(?P<ratio>[-+0-9.eE]+)"
    r"_Step_(?P<step>[-+0-9.eE]+)"
    r"(?:_Index_(?P<index>[-+0-9.eE]+))?"
    r"_SC_(?P<sc>[-+0-9.eE]+)"
    r"_KC_(?P<kc>[-+0-9.eE]+)"
    r"(?:_Nmax_(?P<nmax>[-+0-9.eE]+))?"
)
SAMPLE_PATTERN = re.compile(
    r"assembly_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_entity_number_(?P<entities>\d+)"
)


def parse_legacy_number(text):
    cleaned = str(text).strip()
    cleaned = re.sub(r"([eE][+-]?\d+)\.0$", r"\1", cleaned)
    return float(cleaned)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert UiO_66_from_amorphous samples_summary.csv files into a flat survival-ready CSV."
        )
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root folder containing UiO_66_from_amorphous condition subfolders.",
    )
    parser.add_argument(
        "--target-entities",
        type=float,
        required=True,
        help="Entity threshold used to define the event.",
    )
    parser.add_argument("--output-csv", required=True, help="Output flat CSV path.")
    return parser.parse_args()


def parse_condition_path(raw_path):
    match = PATH_PATTERN.search(raw_path)
    if not match:
        return {}
    parsed = {}
    for name, value in match.groupdict().items():
        if value is None:
            parsed[name] = None
        else:
            parsed[name] = parse_legacy_number(value)
    return parsed


def parse_sample_timestamp(sample_name):
    match = SAMPLE_PATTERN.search(sample_name)
    if not match:
        raise ValueError(f"Could not parse sample timestamp from: {sample_name}")
    date_part = match.group("date")
    time_part = match.group("time")
    return datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H-%M-%S")


def load_summary_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                timestamp = parse_sample_timestamp(row["sample"])
            except ValueError:
                continue
            rows.append(
                {
                    "sample": row["sample"],
                    "timestamp": timestamp,
                    "total_entities": float(row["total_entities"]),
                    "pkl_path": row.get("pkl_path", ""),
                }
            )
    rows.sort(key=lambda item: item["timestamp"])
    return rows


def main():
    args = parse_args()

    root_dir = Path(args.root_dir)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "system",
        "condition_id",
        "source_csv",
        "target_entities",
        "event",
        "time_seconds",
        "first_snapshot_entities",
        "final_entities",
        "snapshot_count",
        "zr_mM",
        "fa_mM",
        "linker_mM",
        "ratio",
        "step",
        "index",
        "sc",
        "kc",
        "nmax",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for summary_csv in sorted(root_dir.rglob("samples_summary.csv")):
            rows = load_summary_rows(summary_csv)
            if not rows:
                continue

            start_time = rows[0]["timestamp"]
            final_time = rows[-1]["timestamp"]
            event_row = next(
                (row for row in rows if row["total_entities"] >= args.target_entities),
                None,
            )

            if event_row is not None:
                event = 1
                time_seconds = (event_row["timestamp"] - start_time).total_seconds()
            else:
                event = 0
                time_seconds = (final_time - start_time).total_seconds()

            parsed = parse_condition_path(rows[0]["pkl_path"])

            writer.writerow(
                {
                    "system": "UiO66_from_amorphous_legacy",
                    "condition_id": summary_csv.parent.name,
                    "source_csv": str(summary_csv),
                    "target_entities": args.target_entities,
                    "event": event,
                    "time_seconds": time_seconds,
                    "first_snapshot_entities": rows[0]["total_entities"],
                    "final_entities": rows[-1]["total_entities"],
                    "snapshot_count": len(rows),
                    "zr_mM": parsed.get("zr"),
                    "fa_mM": parsed.get("fa"),
                    "linker_mM": parsed.get("linker"),
                    "ratio": parsed.get("ratio"),
                    "step": parsed.get("step"),
                    "index": parsed.get("index"),
                    "sc": parsed.get("sc"),
                    "kc": parsed.get("kc"),
                    "nmax": parsed.get("nmax"),
                }
            )


if __name__ == "__main__":
    main()
