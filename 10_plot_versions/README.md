# Plot Versions: Historical vs Shared Snapshot

Files are byte-identical copies, not patched replacements. `historical/` holds
the exact original hashes expected by the October 23 source manifest;
`shared_20261031/` holds revisions found in the shared directory on October 31
(their edits predate that snapshot). The same historical contents also occur
elsewhere in group 07. No previous files were overwritten.

| Pair | Historical location here | Shared revision here | Observed change |
| --- | --- | --- | --- |
| Fig. 6 builder | `historical/figure6/build_figure7_kinetic_window_v2.py` | `shared_20261031/MS_Figure_6/original_code/` | Scatter alpha changed from 0.20 to 0.55; no other line changed |
| Fig. 6 runtime | `historical/common/panel_runtime.py` | `shared_20261031/MS_Figure_6/panel_runtime.py` | Panel B invokes the additional reference-layout helper, also preserved here |
| S7 runtime | `historical/common/panel_runtime.py` | `shared_20261031/SI_Figure_S7/panel_runtime.py` | Selects the seconds renderer instead of raw-u renderer and uses its native export function |

The historical builder hash begins `b2a288c3096c`; the historical runtime hash
begins `15d48882e1c9`. Full hashes and source locations are in the recovered
manifest. The three former `HASH_MISMATCH_REVIEW_REQUIRED` ledger rows now read
`VERSION_VARIANT_ARCHIVED`: `sha256` is the shared revision; `expected_sha256`
still documents the original version, not a claimed match to the new bytes.

## Publication Entry Point

Confirmed by the user on 2026-10-31: **S7 uses algorithm time u**. From the
repository root, run:

```bash
python 10_plot_versions/render_s7.py
```

This uses the archived raw-u numerical renderer, writes `SI_Figure_S7_A-D.svg`
and QA PNGs under `reproduced_local/S7_publication_u/`, and changes only the
historical S8 export filenames to current S7 names. No source data or renderer
bytes are edited. The older generic runtime selecting seconds remains an
archival record, not the release launcher. Fig. 6's adopted file still needs
registration. See `metadata/publication_decisions_20261031.json`.

## Historical Time Conventions

Both S7 renderers are preserved under
`shared_20261031/SI_Figure_S7/original_code/`:

- `export_figure_s7_panels.py`: raw algorithm time `u = CSV time / 0.175`.
- `export_figure_s7_panels_seconds.py`: CSV time directly, already `t_s = 0.175 * u`.

The latter is a room-temperature reference scaling, not an independently
validated absolute framework-growth clock. Do not apply 0.175 twice. It is
retained for historical comparison only, not as a competing publication default.

Historical seconds rendering only, from the repository root:

```bash
python 10_plot_versions/shared_20261031/SI_Figure_S7/original_code/export_figure_s7_panels_seconds.py --data-dir 07_publication_sources/SI_Figure_S7/content_checked_20261023/source_data --output-dir reproduced_local/S7_seconds --qa-dir reproduced_local/S7_seconds/qa
```

The archived generic `panel_runtime.py` files expect their original surrounding
panel package; these partial version archives are not standalone launchers.
They document revisions and dependencies. Fig. 5C's missing numerical analysis
is a separate issue and is not replaced by these scripts.
