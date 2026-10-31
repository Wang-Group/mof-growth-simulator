"""Verify packaged evidence and numeric reproductions, without external paths."""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import matplotlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def serialized_input_policy():
    assert not list(ROOT.rglob('*.docx')), 'Word manuscripts must stay outside the package'
    manifest = json.loads((ROOT / '12_runtime_provenance/source_manifest.json').read_text(encoding='utf-8'))
    allowed = {}
    packaged = set()
    for row in manifest['files']:
        if not row['source_relative'].endswith('.pkl'):
            continue
        assert row['encoding'] == 'identity' and 0 < row['source_bytes'] < 200000
        path = (ROOT / row['release_path']).resolve()
        assert path.is_relative_to(ROOT / '12_runtime_provenance/original')
        assert path.is_file(), path
        allowed[path] = row
        packaged.add(path)
        # Optional materialization is ignored by Git but must retain exact bytes.
        base = ROOT / 'reproduced_local/runtime_source_tree'
        materialized = (base / row['source_relative']).resolve()
        assert materialized.is_relative_to(base)
        allowed[materialized] = row
    assert len(packaged) == 28, 'Expected 28 distinct packaged small seed/host inputs'
    checked = 0
    for path in ROOT.rglob('*.pkl'):
        row = allowed.get(path.resolve())
        assert row is not None, f'Unlisted pickle: {path}'
        assert path.stat().st_size == row['source_bytes'] and digest(path) == row['source_sha256'], path
        checked += 1
    return dict(packaged_seed_host_files=len(packaged), checked_pickles=checked,
                deserialized=False, unlisted_pickles_allowed=False)


def verify(output_dir=None):
    output_dir = Path(output_dir).resolve() if output_dir is not None else ROOT / "reproduced"
    checks = {}
    manifest = pd.read_csv(ROOT / "metadata/source_manifest.csv")
    copied = manifest.loc[manifest.operation == "byte-identical copy"]
    for row in copied.itertuples():
        assert digest(ROOT / row.package_path) == row.source_sha256, row.package_path
    checks["copied_sources_hash_verified"] = len(copied)
    checksums = ROOT / "CHECKSUMS.sha256"
    if checksums.exists():
        lines = checksums.read_text(encoding="utf-8").splitlines()
        for line in lines:
            expected, relative = line.split("  ", 1)
            path = (ROOT / relative).resolve()
            assert ROOT.resolve() in path.parents
            assert digest(path) == expected, relative
        checks["immutable_package_files_hash_verified"] = len(lines)

    root = ROOT / "01_uio66_long_growth"
    first = pd.read_csv(root / "data/first_passage_24.csv").set_index("case_index")
    curves = pd.read_csv(root / "linkage_original/output/uio66_induction_vs_dls_first_passage_curves.csv")
    errors = []
    for case, trace in curves.groupby("case_index"):
        for n in (5, 9, 13, 20, 30, 100, 300, 500, 1000):
            errors.append(abs(trace.loc[trace.entity >= n, "time_s"].iloc[0] - first.loc[case, f"t_entity_{n}_s"]))
    assert len(first) == 24 and len(errors) == 216 and max(errors) == 0
    assert (first.max_entity >= 1000).all()
    for row in pd.read_csv(root / "data/source_manifest.csv").itertuples():
        assert digest(root / "source" / row.file) == row.sha256
    topology = pd.read_csv(root / "data/uio66_entity13_structure_topology_distribution.csv")
    assert len(topology) == 9 and topology.n.sum() == 24 and topology.n.max() == 11
    assert topology.loc[topology.zr6_count == 5, "n"].sum() == 21
    checks["long_growth"] = dict(completed=24, panel_values_matched=216, max_difference=max(errors),
                                  topology_classes=9, largest_class=11,
                                  median_13_to_30_raw_u=float(first.t_13_to_30_s.median() / .175))

    raw = ROOT / "02_dls/data/260316.csv"
    assert digest(raw) == "2364e652ad4bcba27f786b6e7ca9d11ef5cf55a59d2e7036079d57e599b14422"
    p = pd.read_csv(output_dir / "DLS/uio66_dls_gualtieri_parameters.csv").iloc[0]
    expected = dict(nG=2, aN_s=215.53882740461037, bN_s=2.5122048297718993,
                    kG_per_s=.0019279948415726232, r2_alpha=.9726582834248372)
    for key, value in expected.items():
        assert np.isclose(p[key], value, rtol=1e-9, atol=1e-10), (key, p[key], value)
    assert p.n_points == 40 and p.time_start_s == 117 and p.time_end_s == 1170
    # The historical linkage panel and refit must use the same experimental observations.
    old = pd.read_csv(root / "linkage_original/output/uio66_dls_size_gualtieri_overlay_v2_trace.csv")
    new = pd.read_csv(output_dir / "DLS/uio66_dls_gualtieri_processed_trace.csv")
    assert np.array_equal(old.raw_size_nm, new.raw_size_nm)
    assert np.array_equal(old.time_s, new.time_s_with_preparation)
    checks["dls_refit"] = {key: float(p[key]) for key in expected}

    segments = pd.read_csv(ROOT / "03_btb_trajectory/data/trajectory_constant_segments.csv.gz", float_precision="round_trip")
    info = json.loads((ROOT / "03_btb_trajectory/data/export_provenance.json").read_text(encoding="utf-8"))
    assert len(segments) == 148692
    assert segments.row_start.iloc[0] == 0
    assert segments.row_end_inclusive.iloc[-1] == info["raw_rows"] - 1
    assert np.array_equal(segments.row_start.to_numpy()[1:], segments.row_end_inclusive.to_numpy()[:-1] + 1)
    assert (segments.time_end_stored >= segments.time_start_stored).all()
    assert np.all(np.diff(segments.time_start_stored) >= 0)
    checks["btb_segments"] = dict(source_rows=info["raw_rows"], plotted_segments=len(segments),
                                  source_hash_recorded=info["raw_sha256"], time_calibration="pending")

    keys = ["entity", "structure_index"]
    expected_freq = pd.read_csv(ROOT / "04_btb_motifs/data/verified_frequencies.csv").sort_values(keys).reset_index(drop=True)
    actual = pd.read_csv(output_dir / "btb_motif_frequencies.csv").sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected_freq, actual, check_dtype=False, rtol=1e-12)
    for row in actual.itertuples():
        assert (ROOT / f"04_btb_motifs/structures/entity_{row.entity}/unique_structure_{row.structure_index}.mol2").is_file()
    decisions = json.loads((ROOT / "metadata/publication_decisions_20261031.json").read_text(encoding="utf-8"))
    checks["btb_frequencies"] = dict(classes=len(actual), numerical_frequencies_match=True,
                                     candidate_mol2_files_present=True,
                                     structure_confirmation=decisions["Figure5D"]["structures"],
                                     remaining_provenance=decisions["Figure5D"]["remaining"])

    path = next((ROOT / "05_mixed_zr6_zr12/data").glob("*.summary_trace.json"))
    mixed = json.loads(path.read_text(encoding="utf-8"))
    assert len(mixed["replay_runs"]) == 4
    for run in mixed["replay_runs"]:
        initial = run["traces"][0]
        assert [initial[k] for k in ("Zr6_AA", "Zr12_AA", "BDC", "total_entities")] == [37, 37, 98, 172]
        assert run["final_counts"]["Zr12_AA"] == 0 and run["final_counts"]["total_entities"] == 773
    checks["mixed_seed"] = dict(repeats=4, initial_entities=172, final_entities_each=773, final_zr12_each=0)
    repair = pd.read_csv(ROOT / "06_vacancy_repair/data/repair_progress_vs_growth.csv")
    for n, bdc in ((2800, 132), (100001, 584)):
        r = repair.loc[repair.total_entities == n].iloc[0]
        assert r.filled_bdc_sites == bdc and r.filled_zr_sites == 54
        assert r.missing_bdc_sites == 591 and r.missing_zr_sites == 54
        assert np.isclose(r.bdc_fill, bdc / 591) and r.zr_fill == 1
    checks["vacancy_repair"] = dict(denominators=[591, 54], at_2800=[132, 54], at_100001=[584, 54])
    for name in ("Fig3D_raw_u", "FigS7_raw_u", "Fig5A_source_clock", "Fig5D_frequency_check",
                 "FigS21_mixed_counts_check", "FigS23_vacancy_fill_check"):
        for ext in ("png", "svg"):
            assert (output_dir / f"{name}.{ext}").stat().st_size > 100
    checks['small_seed_host_policy'] = serialized_input_policy()
    return dict(status="PASS", python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__, matplotlib=matplotlib.__version__, checks=checks,
                limits=["No KMC production rerun", "No new atom-occupancy reconstruction",
                        "No physical-clock validation", "No final composite image-to-structure validation"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Check fresh renders here instead of supplied reproduced/")
    args = parser.parse_args()
    report = verify(args.output_dir)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
