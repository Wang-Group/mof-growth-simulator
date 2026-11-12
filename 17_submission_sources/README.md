# Modulator Identity, NMR and Scattering Data

The 99 source files retain original bytes, paths and hashes in
`source_manifest.json`.

| Folder | Contents |
|---|---|
| `identity/` | Three CSVs and source notes: 25 MOL constants, selected Chen values and correlation results |
| `nmr/` | Ten selected acquisitions and original/portable spectrum mappings |
| `figure6c/` | Matched-size scattering reference panel at N = 2000, four trajectories per condition |

## NMR

`nmr/portable_mapping.json` uses repository-relative paths. Each acquisition
contains `fid`, `acqus`, `audita.txt`, `pulseprogram`, processed
`1r`/`1i`, `procs`, `auditp.txt` and a spectrum JPG.
Nominal temperatures are 228, 238, 258, 268, 298, 303, 308, 313, 318 and 323 K.
The mapping selects acquisition 2 at 268 K and acquisition 1 at the other
temperatures. Instrument temperature metadata are retained.

Linewidths are the supplied spectrum measurements. The verification command
checks the mapping and exchange-rate arithmetic; it is not an FID fitting entry.

## Scattering

The builder is in
`../07_publication_sources/MS_Figure_6/content_checked_20261023/original_code/build_figures.py`.
N = 2000 spectra and structure statistics are in
`../07_publication_sources/SI_tables_sources_20261023/Table_S22/matched_source_files/`.

## Source and Regression Checks

From the repository root:

```bash
python -m tools.verify_submission_sources
```

This checks 99 source hashes, ten NMR rows, 90 acquisition files and seven
regressions of the supplied constants. The literature-derived Gibbons
correlation is retained as a reference value.

Project data and third-party attribution are described in
[LICENSING.md](../LICENSING.md).
