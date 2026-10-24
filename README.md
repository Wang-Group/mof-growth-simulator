# Verified MOF/MOL Data and Analysis

This branch contains the verified contribution package assembled from the
historical MOF/MOL work on our side: curated numerical data, source snapshots,
structure files, plotting code, and reproducible analysis checks.

**Scope:** a lightweight, independently usable data/analysis subset, not the
complete software release for the latest manuscript. It does not contain the
newer 5090 direct-coordinate campaigns, manuscript/SI files, large raw event
logs, or credentials. Unresolved provenance is explicitly marked rather than
treated as verified.

## Data Folders

| Folder | Contents | Current paper correspondence |
|---|---|---|
| [01_uio66_long_growth](01_uio66_long_growth/) | 24 completed trajectories: first-passage statistics, residence times, representative traces, conditions, historical simulation source, and entity-13 structures | Main Fig. 3D; SI Fig. S7 |
| [02_dls](02_dls/) | Original `260316.csv`, processing and Gualtieri fitting code, parameters, and curves | Main Fig. 3E; SI Section S3.2 |
| [03_btb_trajectory](03_btb_trajectory/) | Compact full step-curve export, original plotting script/PNG, optional exporter, and raw-file fingerprint | Main Fig. 5A; time calibration still pending |
| [04_btb_motifs](04_btb_motifs/) | Historical 284-row source table, 90 motif classes at entity 1-13, frequencies, and candidate MOL2 structures | Main Fig. 5D; exact structure-to-artwork mapping pending |
| [05_mixed_zr6_zr12](05_mixed_zr6_zr12/) | Four regenerated-seed growth records, composition traces, conditions, and initial/final structures | Source family associated with SI Figs. S21-S22 |
| [06_vacancy_repair](06_vacancy_repair/) | Original-vacancy recovery table, initial/N=2800 structures, historical analysis code, and reference figures | Source family associated with SI Figs. S23-S24 |

Within each group, `data/` holds the numerical inputs. `source/`, `scripts/`,
`original/`, and `structures/` preserve available code and structures. Historical
outputs retain their original names; an old `S19` or `Figure3` filename is not a
claim about the current paper's numbering.

## Reproduce and Verify

Use Python 3.11 or newer. The tested versions are in [requirements.txt](requirements.txt).
The existing Anaconda base environment used for this project is sufficient.

```bash
python -m pip install -r requirements.txt
python reproduce.py
python verify_package.py
```

These commands read only local, packaged inputs. They do not run KMC, access
the network/shared drives, modify documents, or need the omitted raw directories.
Outputs are PNG, editable SVG, and numerical CSV files in [reproduced/](reproduced/).
Arial is used for figures; install it to reproduce the original typography.

To draw only one group or preserve the supplied previews:

```bash
python reproduce.py --group dls
python reproduce.py --output-dir reproduced_local
```

The full verifier checks the default `reproduced/` outputs. Re-rendering can
change SVG timestamps without changing the scientific results. `CHECKSUMS.sha256`
covers immutable source data, code, and documentation, not regenerable outputs,
verification reports, or caches.

**Use `reproduce.py` as the entry point.** Archived simulator/exporter scripts
may depend on omitted checkpoints, building blocks, or external paths. Their
presence documents provenance; it does not promise a standalone production KMC
rerun. The original mixed-seed plotting script actually replays simulations and
is not invoked by the lightweight renderer.

## Figure Mapping and Evidence

- [FIGURE_MAP.csv](FIGURE_MAP.csv): data-to-current-figure correspondence, plotting entry points, and confidence limits.
- [metadata/source_manifest.csv](metadata/source_manifest.csv): copied-file hashes, original locations, and documented transformations. Historical paths are provenance only; they are not required at runtime.
- [verification_results.json](verification_results.json): results from the complete local verification workflow.
- [YUHANG_CONFIRMATION_CHECKLIST.md](YUHANG_CONFIRMATION_CHECKLIST.md): outstanding author confirmations and newer missing source families, in Chinese.
- [docs/HANDOFF_20261022.md](docs/HANDOFF_20261022.md): the detailed original handoff note. Its package paths refer to the repository root, not the `docs/` folder.

Figure references follow the final tracked-change view of the October 18 MS and
October 19 SI. Their filenames and hashes are recorded in
[metadata/document_baseline.json](metadata/document_baseline.json); neither
document is included or modified.

## Scientific Limits

- **UiO-66 clock:** historical `_s` tables contain raw algorithm time multiplied by 0.175. The SI S7 replot divides by 0.175 to recover `u`; this is not a new physical-clock calibration. The archived Fig. 3D-E rendering retains the old seconds axis for comparison only.
- **Entity 13:** 24 last-saved snapshots form nine reconstructed topologies, not one unique deterministic nucleus. They are not first-passage structures and are separate from the historical 20-run motif cohort. I03 remains a delayed-growth exception.
- **DLS:** the last two raw columns already include preparation time. The fit uses 40 points spanning 117-1170 s, while the original display ends at 600 s. R-squared = 0.972658 refers to the processed monotonic apparent-progress proxy, not raw diameter or crystallinity. The implementation selects nG from {1, 2, 3}.
- **BTB trajectory:** all step-plot changes are retained in 148,692 constant segments from 5,000,001 original rows. Interior timestamps within constant segments are omitted. The stored source clock is not relabeled as calibrated milliseconds.
- **BTB motifs:** frequencies are historical conditional statistics pooled over 71 encoded conditions, with per-size denominators of 283 or 284; they are not probabilities from the newer response-surface campaign.
- **Defect examples:** the mixed-seed archive is the regenerated four-repeat family. Vacancy-fill fractions use the original 591 BDC and 54 Zr vacancies as denominators. No new occupancy reconstruction from the omitted large checkpoints was performed.

## Branch and Release Status

Branch: `data/verified-contributions-20261024`. It starts with independent Git
history so cloning this branch does not inherit the earlier simulation archive.
It is a standalone contribution release, not a normal merge-back branch. The
original `main` checkout and its uncommitted work remain untouched.

The numerical source package was assembled on 2026-10-22 from repository state
`2a9b1897538430c141c24c17cab746b82a26595c` plus documented shared-drive sources.
The branch preparation date is 2026-10-24. The prior source repository's Git
history remains the authority for earlier authorship; this curated root commit
does not imply sole authorship of archived code.

No new redistribution license, DOI, or complete-paper reproducibility claim is
assigned here. Resolve licensing, the pending time/image mappings, and the newer
calculation sources with the authors before treating this as the final publication
release. Publishing this new branch does not remove old history from `main`.
