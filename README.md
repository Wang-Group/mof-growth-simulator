# MOF/MOL Publication Source Candidate

This candidate combines the previously verified contribution package with the
figure/table-organized sources supplied for the current manuscript and SI.

**Status: GitHub candidate branch, not a complete-paper reproducibility release.**
New response-surface, seeded/surface-growth and finite-bath code and processed
data are now included. On 2026-10-31, the exact corrected engines and eight
models were recovered, plot revisions were separated, and the S21 April 3
source package was added. A second recovery now links small seeds/hosts, exact
code versions and 975 external checkpoint/result pairs; S6's 18 displayed
numbers match the recovered numerical source. Remaining tasks include historical
provenance limits and author release decisions. S6's six code/notebook files and 33
geometries are now packaged with a safe local reproduction entry.
The November 2 Jiageng recovery now links **all 284 Fig. 5D classification rows**
to original run folders and indexed snapshots. All 7,978 recorded assignments
replayed correctly; 3,688 entity 1-13 snapshots are packaged for offline replay.
The existing 90 figure representatives are byte-identical, and the three NAS
repeat-index conflicts are resolved by original-file hashes. Group 15 supersedes
group 14's earlier partial panel D status. **Fig. 5C now has a validated independent
reconstruction in group 16:** unchanged probabilities, 99.74% projected overlap
with historical artwork, and stable grid refinement. The exact historical module
remains unlocated; the new implementation is explicitly labelled, not passed off
as the original. No manuscript artwork was replaced.
**S1's 69/53 counts are author-confirmed** after
literature checking; the historical grouping audit is not an open correction
request. **S2 is confirmed as 23 smaller/thinner outcomes among 30 publications**;
the updated MS and SI agree. **Table S8 is closed:** the author confirmed that the
existing table is the original recorded turbidity-time data; no additional raw
record is requested (status updated 2026-11-01). Manuscript/SI files,
large binary trajectories and credentials are not included.

Start with [the publication checklist](docs/PUBLICATION_RELEASE.md), then use
[the code entry-point index](docs/CODE_INDEX.md) and
[the source inventory](metadata/publication_sources.csv). Every inventory row
distinguishes copied evidence from excluded, external or changed files. Finding
a file is not the same as reproducing its full scientific workflow.

## Data Folders

| Folder | Contents | Current paper correspondence |
|---|---|---|
| [01_uio66_long_growth](01_uio66_long_growth/) | 24 completed trajectories: first-passage statistics, residence times, representative traces, conditions, historical simulation source, and entity-13 structures | Main Fig. 3D; SI Fig. S7 |
| [02_dls](02_dls/) | Original `260316.csv`, processing and Gualtieri fitting code, parameters, and curves | Main Fig. 3E; SI Section S3.2 |
| [03_btb_trajectory](03_btb_trajectory/) | Compact full step-curve export, original plotting script/PNG, optional exporter, and raw-file fingerprint | Main Fig. 5A; time calibration still pending |
| [04_btb_motifs](04_btb_motifs/) | Historical 284-row source table, 90 motif classes at entity 1-13, frequencies, and MOL2 structures | Main Fig. 5D; unchanged originals, now linked to raw runs through group 15 |
| [05_mixed_zr6_zr12](05_mixed_zr6_zr12/) | Four April 10 rerun records, composition traces, conditions, and initial/final structures | Historical comparison family; do not substitute for the April 3 S21 source in group 11 |
| [06_vacancy_repair](06_vacancy_repair/) | Original-vacancy recovery table, initial/N=2800 structures, historical analysis code, and reference figures | Source family associated with SI Figs. S23-S24 |
| [07_publication_sources](07_publication_sources/) | 670 byte-verified code/data/configuration copies, retaining supplied figure/table paths and version labels | Current figure/table source bundles; panel-specific limits apply |
| [08_entropy_inputs](08_entropy_inputs/) | 13 unchanged tables, six code/notebook files, 33 geometries and a safe S6 entry | All six geometry mappings and 18 S6 values reproduced locally; no batch writers executed |
| [09_recovered_runtime](09_recovered_runtime/) | Two corrected engines and eight molecular models, recovered byte-identically from the 5090 | Exact UiO-66 engine matches all six recorded runs; full campaign replay still requires upstream inputs |
| [10_plot_versions](10_plot_versions/) | Historical and shared-directory revisions of Fig. 6/S7 plotting code, including raw-u and seconds S7 renderers | Three former hash discrepancies now have separate, documented versions |
| [11_s21_april03](11_s21_april03/) | April 3 Zr6/Zr12 saved trajectories, plotting code, artwork and an explicitly separated alternative source | Shared-source README identifies the main variant as S21; final document adoption still requires author confirmation |
| [12_runtime_provenance](12_runtime_provenance/) | 206 recovered source locations, small seed/host inputs, 197 exact code/seed references, 1327 lineage rows and remote checkpoint hashes | Separate campaign families; six S7 results and four sampled historical results linked to seed/RNG/parameters/code |
| [13_table_source_audit](13_table_source_audit/) | Original S1 evidence, historical final-view SI extracts, S6 comparison and current status note | S1/S2 counts confirmed; S8 closed; S6 sources and local reproduction complete |
| [14_fig5_source_followup](14_fig5_source_followup/) | Earlier NAS/5090 sources and November 1 search records | Historical partial panel D audit; panel C original module still unlocated |
| [15_fig5d_trajectory_lineage](15_fig5d_trajectory_lineage/) | 284 run mappings, 7978 indexed snapshots, 3688 packaged figure-size snapshots, original indices and source versions | Current Fig. 5D provenance; fixed-representative classifier replay and four raw-block hash spot checks |
| [16_fig5c_reconstruction](16_fig5c_reconstruction/) | Identical condition table, independent source, historical SVG geometry, portable entry and convergence report | Fig. 5C reproducible reconstruction; projected historical consistency, not original 3D-mesh recovery |

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
python verify_publication.py --report publication_verification.json
python tools/reproduce_entropy.py
python -m tools.audit_fig5d_lineage
python 16_fig5c_reconstruction/render.py
python -m tools.audit_fig5c
python -m unittest discover -s tests -v
```

These commands read only local, packaged inputs. They do not run KMC, access
the network/shared drives, modify documents, or need the omitted raw directories.
Outputs are PNG, editable SVG, and numerical CSV files in [reproduced/](reproduced/).
Arial is used for figures; install it to reproduce the original typography.
`reproduce.py` covers groups 01-06 only. `verify_publication.py` verifies the new
source hashes, independently recomputes Fig. 5B, checks S25 and smoke-tests the
pure event selector. It now verifies the recovered sources, all three mixed-seed
variant means, and all 18 engine/runner/selector hash references in six UiO-66
result records. The group 12/13 audit also checks source decompression, saved
checkpoint lineage, exact seed/RNG/parameter matches and S1/S6 numerical evidence.
Fig. 5C verification recalculates four mesh grids and compares historical projected
boundaries. Its standalone commands write to `reproduced_local/`; frozen checks
and previews are in group 16. This is not statistical uncertainty estimation.
A passing
check is deliberately not labelled publication-ready.

To draw only one group or preserve the supplied previews:

```bash
python reproduce.py --group dls
python reproduce.py --output-dir reproduced_local
python verify_package.py --output-dir reproduced_local
```

The full verifier checks `reproduced/` unless `--output-dir` is supplied. Re-rendering can
change SVG timestamps without changing the scientific results. `CHECKSUMS.sha256`
covers immutable source data, code, and documentation, not regenerable outputs,
verification reports, or caches.

**Use `reproduce.py` as the entry point.** Archived simulator/exporter scripts
may depend on omitted checkpoints, building blocks, or external paths. Their
presence documents provenance; it does not promise a standalone production KMC
rerun. The original mixed-seed plotting script actually replays simulations and
is not invoked by the lightweight renderer.

The new S21 source can be drawn without simulation using the separate plotting-only
helper documented in [group 11](11_s21_april03/README.md). This does not change
`reproduce.py --group mixed`, which intentionally still draws the April 10 rerun.

**The confirmed S7 publication coordinate is algorithm time u.** Run
`python 10_plot_versions/render_s7.py` for standalone S7 A-D SVGs and QA PNGs.
The seconds renderer is retained only as a historical version.

## Figure Mapping and Evidence

- [FIGURE_MAP.csv](FIGURE_MAP.csv): data-to-current-figure correspondence, plotting entry points, and confidence limits.
- [metadata/source_manifest.csv](metadata/source_manifest.csv): copied-file hashes, original locations, and documented transformations. Historical paths are provenance only; they are not required at runtime.
- [metadata/publication_sources.csv](metadata/publication_sources.csv): current source selection, hashes, exclusions and evidence scope for the added figure/table material.
- [metadata/recovered_sources_20261031.json](metadata/recovered_sources_20261031.json): 42 byte-identical additions with original host/project or shared-directory paths; no login credentials.
- [metadata/publication_decisions_20261031.json](metadata/publication_decisions_20261031.json): S7 time coordinate, Fig. 5D structure status, S1/S2/S8 confirmations and the November 1 NAS follow-up; confirmations are distinguished from independent checks.
- [metadata/runtime_table_audit_20261031.json](metadata/runtime_table_audit_20261031.json): latest seed/code/checkpoint and S1/S2/S6/S8 audit, including explicit limits and document hashes.
- [publication_verification.json](publication_verification.json): new numerical checks and exact-hash coverage of executed code recorded in six UiO-66 results.
- [verification_results.json](verification_results.json): results from the complete local verification workflow.
- [YUHANG_CONFIRMATION_CHECKLIST.md](YUHANG_CONFIRMATION_CHECKLIST.md): outstanding author confirmations and newer missing source families, in Chinese.
- [docs/HANDOFF_20261022.md](docs/HANDOFF_20261022.md): the detailed original handoff note. Its package paths refer to the repository root, not the `docs/` folder.

The last document review baseline, including the changed MS file with its unchanged
October 18 filename, is [recorded here](metadata/release_document_baseline.json).
Insertions were included and deletions excluded when reading the documents.
The older [baseline](metadata/document_baseline.json) is retained for history;
neither document is included or modified. Group 13 preserves the October 31
selected-table baseline unchanged. The November 1 follow-up found a newer MS
and S2-updated SI. After the author retained the checked study and updated the
MS, both documents state 23 smaller/thinner outcomes among 30 base-additive
publications. The discrepancy is closed; current hashes and the retention
decision are recorded in the existing decisions file. S6's six species and
18 values were independently recalculated using recovered NAS notebooks and
mapped geometries; these sources and the reproduction entry are now packaged. Neither follow-up
replaces a complete document review or the local verifier's narrower scope.

## Scientific Limits

- **UiO-66 clock:** historical `_s` tables contain raw algorithm time multiplied by 0.175. The confirmed S7 publication entry divides by 0.175 to recover `u`. Group 10 preserves the seconds renderer only as history. This is not a new physical-clock calibration and does not automatically relabel other figures.
- **Entity 13:** 24 last-saved snapshots form nine reconstructed topologies, not one unique deterministic nucleus. They are not first-passage structures and are separate from the historical 20-run motif cohort. I03 remains a delayed-growth exception.
- **DLS:** the last two raw columns already include preparation time. The fit uses 40 points spanning 117-1170 s, while the original display ends at 600 s. R-squared = 0.972658 refers to the processed monotonic apparent-progress proxy, not raw diameter or crystallinity. The implementation selects nG from {1, 2, 3}.
- **BTB trajectory:** all step-plot changes are retained in 148,692 constant segments from 5,000,001 original rows. Interior timestamps within constant segments are omitted. The stored source clock is not relabeled as calibrated milliseconds.
- **Recovered BTB run:** the Fig. 5A pickle matches archived run Index_2 by SHA-256. Its run-folder source saves raw `timing`, without multiplying by 0.1. The later plotter multiplies by 0.2405; this recovers the numeric plotting convention, not a new physical calibration. See [the recovered source note](03_btb_trajectory/source/README.md).
- **BTB motifs:** frequencies are historical conditional statistics pooled over 71 encoded conditions, with per-size denominators of 283 or 284 at entity 1-13; they are not probabilities from the newer response-surface campaign. The recovered extractor selects the last occurrence of each size before the first maximum, not first-passage structures. Group 15 reproduces historical radial-histogram class assignments, not graph isomorphism or exact RNG histories.
- **Defect examples:** three distinct mixed-seed variants are preserved: April 3 `Zr6_Zr12_mix`, April 3 `initialize_structures1`, and the April 10 rerun. Their stored mean-Zr12-zero times differ (0.701327, 0.306265, and 0.270601 s); they must not be relabeled or pooled as the same cohort. Vacancy-fill fractions retain the original 591 BDC and 54 Zr vacancies as denominators. No new occupancy reconstruction from omitted checkpoints was performed.

## Branch and Release Status

Branch: `codex/publication-release-20261029`, based on the independent-history
`data/verified-contributions-20261024` branch. Neither that original branch nor
the original `main` checkout and its uncommitted work were changed. This is not
a normal merge-back branch and must not replace the full historical archive.

The numerical source package was assembled on 2026-10-22 from repository state
`2a9b1897538430c141c24c17cab746b82a26595c` plus documented shared-drive sources.
The publication candidate was prepared on 2026-10-29 and extended on 2026-10-31
with approximately 14.3 MiB in the first source integration, then another
5.4 MiB of runtime source payload plus provenance/table metadata. The prior source repository's Git
history remains the authority for earlier authorship; this curated root commit
does not imply sole authorship of archived code.

No new redistribution license, DOI, or complete-paper reproducibility claim is
assigned here. Resolve licensing, the pending time/image mappings, and the newer
calculation sources with the authors before treating this as the final publication
release. Publishing this new branch does not remove old history from `main`.
