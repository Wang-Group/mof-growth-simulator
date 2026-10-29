from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"C:\Users\SYH\Documents\Codex\2026-05-11\gitlab-ssh-gitlab-git-xmu-edu")
OUTDIR = ROOT / "outputs" / "zr_core_radius_stats_verified"
SOURCE_DIR = Path(
    r"D:\课题相关文献\课题\MOL生长模拟\论文\data1\Other-MOFs-Nucleation\mnt\syh\MOF_growth_data\_combined_uio66_e1_to_50_pairtol_0p2\unique_structures"
)
EXISTING_ALL_CSV = ROOT / "outputs" / "zr_core_radius_stats" / "zr_core_outer_size_all.csv"
ZR_RADIUS_NM = 0.16


@dataclass
class Mol2Record:
    entity_count: int
    file: str
    zr_atom_count: int
    zr6_cluster_count: int
    max_zr_center_distance_nm: float
    max_zr_core_outer_distance_nm: float


def parse_mol2_zr_coords(path: Path) -> list[tuple[float, float, float]]:
    coords: list[tuple[float, float, float]] = []
    in_atom_block = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom_block = True
            continue
        if in_atom_block and line.startswith("@<TRIPOS>"):
            break
        if not in_atom_block or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        atom_name = parts[1]
        atom_type = parts[5]
        if atom_name == "Zr" or atom_type == "Zr":
            coords.append((float(parts[2]), float(parts[3]), float(parts[4])))
    return coords


def max_pair_distance_nm(coords_angstrom: list[tuple[float, float, float]]) -> float:
    if len(coords_angstrom) < 2:
        return 0.0
    max_sq = 0.0
    for a, b in combinations(coords_angstrom, 2):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        dist_sq = dx * dx + dy * dy + dz * dz
        if dist_sq > max_sq:
            max_sq = dist_sq
    return (max_sq ** 0.5) / 10.0


def build_records() -> list[Mol2Record]:
    records: list[Mol2Record] = []
    for entity_count in range(1, 21):
        entity_dir = SOURCE_DIR / f"entity_{entity_count:03d}"
        files = sorted(entity_dir.glob("*.mol2"))
        for file_path in files:
            zr_coords = parse_mol2_zr_coords(file_path)
            zr_atom_count = len(zr_coords)
            max_center = max_pair_distance_nm(zr_coords)
            max_outer = max_center + 2 * ZR_RADIUS_NM if zr_atom_count > 0 else 0.0
            records.append(
                Mol2Record(
                    entity_count=entity_count,
                    file=file_path.name,
                    zr_atom_count=zr_atom_count,
                    zr6_cluster_count=zr_atom_count // 6 if zr_atom_count else 0,
                    max_zr_center_distance_nm=round(max_center, 4),
                    max_zr_core_outer_distance_nm=round(max_outer, 4),
                )
            )
    return records


def build_summary(df_filtered: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for entity_count, group in df_filtered.groupby("entity_count", sort=True):
        sizes = group["max_zr_core_outer_distance_nm"].astype(float).sort_values()
        rows.append(
            {
                "entity_count": int(entity_count),
                "n_structures": int(len(group)),
                "min_size_nm": round(float(sizes.min()), 4),
                "max_size_nm": round(float(sizes.max()), 4),
                "mean_size_nm": round(float(sizes.mean()), 4),
                "median_size_nm": round(float(sizes.median()), 4),
                "std_size_nm": round(float(sizes.std(ddof=1)) if len(sizes) > 1 else 0.0, 4),
                "unique_sizes_nm": ";".join(f"{v:.4f}" for v in pd.unique(sizes)),
            }
        )
    return pd.DataFrame(rows)


def compare_with_existing(new_df: pd.DataFrame) -> str:
    existing = pd.read_csv(EXISTING_ALL_CSV)
    merge_cols = [
        "entity_count",
        "file",
        "zr_atom_count",
        "zr6_cluster_count",
        "max_zr_center_distance_nm",
        "max_zr_core_outer_distance_nm",
    ]
    new_sorted = new_df[merge_cols].sort_values(["entity_count", "file"]).reset_index(drop=True)
    existing_sorted = existing[merge_cols].sort_values(["entity_count", "file"]).reset_index(drop=True)

    if new_sorted.equals(existing_sorted):
        return (
            "Verified: the recomputed entity_001 to entity_020 dataset matches the existing "
            "zr_core_outer_size_all.csv exactly, including the single zr_atom_count = 0 row in entity 1."
        )

    diffs = []
    if len(new_sorted) != len(existing_sorted):
        diffs.append(f"Row count mismatch: new={len(new_sorted)} existing={len(existing_sorted)}")
    merged = new_sorted.merge(
        existing_sorted,
        on=["entity_count", "file"],
        how="outer",
        suffixes=("_new", "_old"),
        indicator=True,
    )
    for _, row in merged.iterrows():
        if row["_merge"] != "both":
            diffs.append(f"{row['entity_count']} {row['file']}: presence mismatch ({row['_merge']})")
            continue
        for col in [
            "zr_atom_count",
            "zr6_cluster_count",
            "max_zr_center_distance_nm",
            "max_zr_core_outer_distance_nm",
        ]:
            new_val = row[f"{col}_new"]
            old_val = row[f"{col}_old"]
            if pd.isna(new_val) and pd.isna(old_val):
                continue
            if float(new_val) != float(old_val):
                diffs.append(f"{row['entity_count']} {row['file']} {col}: new={new_val} old={old_val}")
    return "Differences found:\n" + "\n".join(diffs[:50])


def make_plot(df_filtered: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 14,
            "axes.titlesize": 26,
            "axes.labelsize": 22,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

    fig, ax = plt.subplots(figsize=(12.5, 7.6), dpi=200)

    box_data = [
        df_filtered.loc[df_filtered["entity_count"] == entity, "max_zr_core_outer_distance_nm"].tolist()
        for entity in range(1, 21)
    ]
    positions = list(range(1, 21))
    ax.boxplot(
        box_data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#c7d6eb", "edgecolor": "#4169e1", "linewidth": 1.4},
        whiskerprops={"color": "#4169e1", "linewidth": 1.4},
        capprops={"color": "#4169e1", "linewidth": 1.4},
        medianprops={"color": "#0f172a", "linewidth": 1.8},
    )

    ax.plot(
        summary_df["entity_count"],
        summary_df["mean_size_nm"],
        color="#0f9a6c",
        marker="o",
        markersize=5.5,
        linewidth=2.0,
    )

    ax.set_xlim(0.5, 20.5)
    ax.set_xticks(positions)
    ax.set_xlabel("Entity count")
    ax.set_ylabel("Max Zr-core outer distance (nm)")
    ax.set_title("Zr-core simulated size distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.22, linewidth=1)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("#2f2f2f")
    ax.tick_params(width=1.6, length=6)

    fig.tight_layout()
    fig.savefig(OUTDIR / "zr_core_size_distribution_verified.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTDIR / "zr_core_size_distribution_verified.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    records = build_records()
    df_all = pd.DataFrame([record.__dict__ for record in records])
    df_filtered = df_all[df_all["zr_atom_count"] > 0].copy()
    summary_df = build_summary(df_filtered)

    df_all.to_csv(OUTDIR / "zr_core_outer_size_all_verified.csv", index=False)
    df_filtered.to_csv(OUTDIR / "zr_core_outer_size_all_verified_filtered.csv", index=False)
    summary_df.to_csv(OUTDIR / "zr_core_outer_size_summary_verified.csv", index=False)

    verification_text = compare_with_existing(df_all)
    verification_lines = [
        f"Source directory: {SOURCE_DIR}",
        "Definition: max Zr-core outer distance = max(Zr-Zr center distance) + 2 * Zr atomic radius",
        f"Zr atomic radius used: {ZR_RADIUS_NM:.2f} nm",
        "",
        verification_text,
    ]
    (OUTDIR / "verification_report.txt").write_text("\n".join(verification_lines), encoding="utf-8")

    make_plot(df_filtered, summary_df)


if __name__ == "__main__":
    main()
