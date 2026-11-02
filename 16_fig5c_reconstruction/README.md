# Figure 5C: Validated Independent Reconstruction

**Status: reproducible numerical/plotting route validated on 2026-11-02.**
This is not the recovered historical `analyze_figure6_current_parameter_phase_v1.py`.
No MS/SI or author artwork was changed. Historical files remain in their original
groups. The old-module search is a provenance limitation, not missing numerical
input or a blocker for this explicitly identified reconstruction.

## Run Offline

From the repository root, with the dependencies in `requirements.txt`:

```bash
python 16_fig5c_reconstruction/render.py
python -m tools.audit_fig5c
```

The first command writes Arial PNG/SVG and a method record to
`reproduced_local/Fig5C/`. The second writes a comparison figure, convergence
table, exact boundary roots and JSON report to `reproduced_local/Fig5C_validation/`.
The checked-in `validation/` folder is the dated validation snapshot. No network,
simulation, external archive, or private workstation path is needed.

## Data And Versions

- `data/`: byte-identical condition table, SHA-256
  `866fee5ec5a3b0e4f80bc776b4f1880bdfb0f0694cc0bc0e02ac0ddb95e31763`.
  It matches group 07. Only the 1260 `btb_mol` rows are used, with 32 launched
  traces per condition. `P = target_count / trace_count` exactly matches the
  archived probability column. The probability classes contain 551 / 40 / 669 rows.
- `source/`: unchanged NAS November 1 independent script and its README.
  Its original default was 34 x 38 x 22. That README describes that version,
  not the portable entry's selected finer grid.
- `render.py`: portable entry with a 67 x 75 x 43 grid, yielding 32180 triangles.
  It calls the unchanged numerical implementation, explicitly sets the grid,
  fixes the middle legend interval to strict inequalities, and writes provenance.
  To reproduce the NAS grid, pass `--grid 34 38 22`.
- `reference/`: losslessly compressed historical C SVG and a path-only extract
  from the author's Figure5.svg. Source locations, hashes and transformations
  are in `source_manifest.json`. The 25 MB author composite is not duplicated.
- `validation/`: projected-boundary comparison, four-grid convergence results,
  root table, report, and a reconstructed-panel preview. Not automatically
  adopted manuscript artwork.

## Numerical Method

The complete 10 x 14 x 9 data grid is interpolated trilinearly in
`log10(rho_off)`, `log10(M/BTB)` and linear `s_close`. No smoothing, fitting,
missing-value imputation or extrapolation is used. Marching tetrahedra creates
the P=0.5 mesh; intersected edges are refined against the fixed interpolator.
All conditions determine the surface. Scatter display uses all 40 transition
conditions plus a deterministic 20% low/high sample (seed 20260901).
This is a kinetic outcome boundary, not a thermodynamic phase boundary.

## Validation Results

The historical SVG contains 14488 triangles. Its paths match the author
composite before page-layout transforms within 0.000752 SVG pt. A perspective
camera fitted to 45 axis/grid control points, not the surface, has maximum
reprojection error below 0.000001 pt. Thus the comparison is not obtained by
warping one surface to fit the other.

| Refinement Grid | Triangles | Projected IoU With Historical Surface | 95th-Percentile Centroid Error in log10(rho) |
|---|---:|---:|---:|
| 17 x 19 x 11 | 1774 | 99.36% | 0.04039 |
| 34 x 38 x 22 | 7888 | 99.67% | 0.01200 |
| **67 x 75 x 43 (entry default)** | **32180** | **99.74%** | **0.00295** |
| 133 x 149 x 85 | 129363 | 99.74% | 0.00072 |

At the entry default, the historical/new projected-boundary separation is
0.177 pt at the 95th percentile and 0.396 pt maximum. The default and finest
meshes have 99.88% projected IoU. Mask resolution was also varied from 4 to 16
pixels per SVG pt; metrics and explicit engineering checks are in the JSON.
These are drawing-level consistency checks, not statistical equivalence tests.

Data trends are preserved rather than refitted: P is nonincreasing with rho_off
on all 126 sampled M/BTB x s_close slices. Average P increases with M/BTB and
s_close overall, but some adjacent contrasts reverse (23 for M/BTB, 11 for
s_close); no strict monotonic claim is made for those two parameters.

## Limits

Only one historical 2D projection is available; this does not establish exact
equality of all original 3D vertices or recover the historical algorithm.
The selected mesh's unweighted triangle-centroid rho error is about 0.68% at the
95th percentile, but the maximum bound is 7.98% locally (0.64% of the full
log-rho axis). The maximum centroid P error is 0.0523; do not infer high-precision
thresholds from a display triangle. Use `render.exact_roots` on the fixed
interpolator instead; the exported sampled-coordinate root table uses that route.
The root convention is the rightmost bracketed P=0.5 point (the upper end of
an exactly flat P=0.5 interval), not an assumption of a unique root everywhere.
Refinement reduces both typical and maximum centroid errors. These errors do
not include binomial uncertainty from 32 traces or physical-model uncertainty.

The historical module remains unlocated. The independent implementation,
validated scope, remaining provenance limit and manuscript-adoption status must
not be conflated. No bootstrap or new KMC campaign was run.
