# Submission Source Supplement

Imported on 2026-11-02 from the shared project, without editing MS/SI. The
99 source files total 8,578,932 bytes; each source path, size and SHA-256 is in
`source_manifest.json`. Copies preserve original bytes, including source notes.

| Folder | Contents | Use |
|---|---|---|
| `identity/` | Three original CSVs and their source README | SI identity comparisons; 25 MOL constants, selected Chen values and correlation results |
| `nmr/` | Original mapping JSON/notes, portable mapping, ten selected acquisitions | Table S4 row-to-spectrum correspondence |
| `figure6c/` | Adopted N=2000 reference panel in PNG/SVG | Fig. 6C, four trajectories per condition |

## Verify Offline

```bash
python -m tools.verify_submission_sources
```

Run from the repository root. No NAS, SSH or Word installation is required.
The check verifies all 99 source hashes, ten NMR rows and 90 acquisition files,
the source/reference document registration, and seven regression comparisons.
The retained Gibbons literature-derived correlation row is not independently
recalculated by this check. Regressions of supplied constants do not refit the
underlying adsorption experiments.

`nmr/portable_mapping.json` uses repository-relative paths. Each acquisition
contains `fid`, `acqus`, `audita.txt`, `pulseprogram`, processed `1r`/`1i`,
`procs`, `auditp.txt`, and a spectrum JPG. Nominal temperatures are 228, 238,
258, 268, 298, 303, 308, 313, 318 and 323 K. The author selected acquisition 2
at 268 K and acquisition 1 at the other temperatures. Instrument temperature
metadata are retained, not replaced by the nominal SI temperatures.

Table S4 linewidths remain author/software measurements; verification checks
the mapping and displayed exchange-rate arithmetic, not independent FID fitting.
The 298 K JPG is a separate supplied export, not claimed to be the exact SI
S50 bitmap. The original mapping retains its earlier document hash; the current
reviewed MS/SI hashes are recorded separately in the root document baseline.

The Fig. 6C builder and N=2000 spectra/structure statistics are already in group
07 (`MS_Figure_6/content_checked_20261023/original_code/build_figures.py` and
`SI_tables_sources_20261023/Table_S22/matched_source_files/`). The source manifest
registers the current MS composite-image hash; the manuscript itself is excluded.

`tools/import_submission_sources.py --shared-project <project-root>` documents
the hash-pinned import and is only needed to repeat collection from the original
shared drive. Ordinary analysis uses the packaged copies.

Project-produced data are CC BY 4.0; original code is MIT. Literature-derived
values and third-party materials retain their original attribution and rights;
see the root `LICENSING.md`.
