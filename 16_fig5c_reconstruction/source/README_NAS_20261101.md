# Figure 5C independent numerical rebuild

Run from this directory with Python containing `numpy`, `pandas`, `scipy`, and `matplotlib`:

```powershell
python rebuild_figure5c_standalone.py
```

This single-file replacement reads `input_data/common_response_surface_v2_final_condition_summary.csv`, filters `system == btb_mol`, calculates `P = target_count / trace_count` for 1260 conditions (32 launched traces each), and writes `Figure5C_standalone_transparent_outline_larger_text.svg`, `.png`, and `_method.json`. The plotted points have 38% fill opacity and 82% opacity outlines. The default text scale is 1.2, with axis tick values enlarged further to 11.3 pt. Previous outputs remain in this folder. The copied source table has SHA-256 `866fee5ec5a3b0e4f80bc776b4f1880bdfb0f0694cc0bc0e02ac0ddb95e31763`, identical to the archived table in `MS_Figure_5/content_checked_20261023/source_data/`.

The input must be a complete 10 x 14 x 9 condition grid without duplicate or missing coordinates; no missing values are filled. The `P = 0.5` surface uses trilinear interpolation in log10(rho_off), log10(M/BTB), and linear s_close, followed by marching tetrahedra on a 34 x 38 x 22 fine grid. Each edge intersection is refined against the original trilinear interpolator. Interpolation is confined to the sampled domain; if no boundary occurs, the program fails instead of inventing one. All 1260 conditions determine the surface. The display uses all 40 transition conditions and a reproducible 20% sample of low/high conditions (seed 20260901). The surface is a kinetic formation boundary, not a thermodynamic phase boundary.

Validation: the 1260 probabilities exactly agree with the archived `persistent_assembly_probability` column; the original panel manifest reports the same 551/40/669 probability-class counts. The rebuilt surface has 7888 triangles, and its vertices return to P=0.5 within 2.5e-8 under the input interpolator. The output SVG is vector-only and contains no embedded raster image. The old manifest reports 7414 vertices and 14488 faces, so this reconstruction is not a byte-identical regeneration of the historical mesh. The copied CSV summarizes simulation outcomes rather than individual trajectory logs. Source files and existing manuscript artwork remain unchanged.
