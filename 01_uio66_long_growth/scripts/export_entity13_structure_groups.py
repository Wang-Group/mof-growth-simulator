from __future__ import annotations

import csv
import shutil
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PACKAGE_ROOT / "data"
CASES_CSV = OUTPUT_DIR / "uio66_entity13_structure_cases.csv"
GROUP_DIR = PACKAGE_ROOT / "structures" / "entity13_groups"


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


def read_cases() -> list[dict[str, str]]:
    with CASES_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    rows = read_cases()
    GROUP_DIR.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["topology_id"], []).append(row)

    ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    index_rows: list[dict[str, str]] = []
    root_lines = [
        "# UiO-66 entity-13 structure groups",
        "",
        "These files are grouped by reconstructed SBU-linker topology from the 24 completed long-growth trajectories.",
        "",
        "Important limitation: the entity-13 snapshots are mol2 files with zero bond records and no corresponding pkl files. The grouping therefore uses geometry-reconstructed BDC-Zr6 connections.",
        "",
        "| Group | Topology ID | n | Repeats | Zr6/BDC | Edges | Mono/bridging BDC | Cycle rank |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]

    for group_index, (topology_id, group_rows) in enumerate(ordered_groups, start=1):
        first = group_rows[0]
        folder_name = (
            f"group_{group_index:02d}_n{len(group_rows):02d}_{topology_id}_"
            f"Zr{first['zr6_count']}_BDC{first['bdc_count']}_"
            f"E{first['edge_count']}_C{first['cycle_rank']}"
        )
        folder = GROUP_DIR / safe_name(folder_name)
        folder.mkdir(parents=True, exist_ok=True)

        repeats = ",".join(row["repeat_id"] for row in group_rows)
        root_lines.append(
            "| {group} | `{topology}` | {n} | {repeats} | {zr}/{bdc} | {edges} | {mono}/{bridging} | {cycle} |".format(
                group=f"group_{group_index:02d}",
                topology=topology_id,
                n=len(group_rows),
                repeats=repeats,
                zr=first["zr6_count"],
                bdc=first["bdc_count"],
                edges=first["edge_count"],
                mono=first["mono_linker_count"],
                bridging=first["bridging_linker_count"],
                cycle=first["cycle_rank"],
            )
        )

        group_lines = [
            f"# group_{group_index:02d}: {topology_id}",
            "",
            f"- Count: `{len(group_rows)}` / 24",
            f"- Repeats: `{repeats}`",
            f"- Zr6/BDC: `{first['zr6_count']}/{first['bdc_count']}`",
            f"- Reconstructed edges: `{first['edge_count']}`",
            f"- Mono/bridging BDC: `{first['mono_linker_count']}/{first['bridging_linker_count']}`",
            f"- Cycle rank: `{first['cycle_rank']}`",
            "",
            "Files:",
            "",
        ]

        for row_index, row in enumerate(group_rows):
            repeat_id = row["repeat_id"]
            source = Path(row["mol2_path"])
            copied_name = (
                f"I{repeat_id}_{topology_id}_Zr{row['zr6_count']}_"
                f"BDC{row['bdc_count']}_E{row['edge_count']}_"
                f"C{row['cycle_rank']}.mol2"
            )
            destination = folder / safe_name(copied_name)
            shutil.copy2(source, destination)
            copied_relative = destination.relative_to(PACKAGE_ROOT).as_posix()
            try:
                source_relative = source.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                source_relative = source.name
            row["mol2_path"] = copied_relative

            if row_index == 0:
                representative = folder / f"representative_I{repeat_id}_{topology_id}.mol2"
                shutil.copy2(source, representative)
                group_lines.append(f"- `representative_I{repeat_id}_{topology_id}.mol2`")

            group_lines.append(f"- `{destination.name}`")
            index_rows.append(
                {
                    "group": f"group_{group_index:02d}",
                    "group_folder": folder.relative_to(PACKAGE_ROOT).as_posix(),
                    "repeat_id": repeat_id,
                    "topology_id": topology_id,
                    "zr6_count": row["zr6_count"],
                    "bdc_count": row["bdc_count"],
                    "edge_count": row["edge_count"],
                    "mono_linker_count": row["mono_linker_count"],
                    "bridging_linker_count": row["bridging_linker_count"],
                    "cycle_rank": row["cycle_rank"],
                    "copied_mol2": copied_relative,
                    "source_mol2": source_relative,
                }
            )

        write_text(folder / "README.md", "\n".join(group_lines) + "\n")

    root_lines.extend(
        [
            "",
            "Suggested viewing order: open the representative mol2 in each group first, then inspect the individual repeats within the same group if needed.",
            "",
        ]
    )
    write_text(GROUP_DIR / "README.md", "\n".join(root_lines))

    with (GROUP_DIR / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "group",
            "group_folder",
            "repeat_id",
            "topology_id",
            "zr6_count",
            "bdc_count",
            "edge_count",
            "mono_linker_count",
            "bridging_linker_count",
            "cycle_rank",
            "copied_mol2",
            "source_mol2",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index_rows)

    # Replace machine-specific raw paths in the committed case table with the
    # corresponding portable paths inside this package.
    with CASES_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Grouped structures written to: {GROUP_DIR}")
    print(f"Groups: {len(ordered_groups)}; structures copied: {len(index_rows)}")


if __name__ == "__main__":
    main()
