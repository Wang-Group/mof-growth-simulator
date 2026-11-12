# Plotting Versions and Time Coordinates

Source files retain their original bytes. `historical/` contains the original
plotting versions; `shared_20261031/` contains the corresponding shared-folder
revisions. Source hashes distinguish these versions.

| Plotting component | Version difference |
|---|---|
| Matched-size BTB builder | Scatter alpha changed from 0.20 to 0.55 |
| Matched-size BTB runtime | Added a reference-layout helper |
| UiO-66 ensemble runtime | Selected the stored-seconds renderer instead of the raw-u renderer |

## UiO-66 Ensemble Plot

From the repository root:

```bash
python 10_plot_versions/render_s7.py
```

This uses raw algorithm time `u = stored CSV time / 0.175` and writes SVG/PNG
panels to `reproduced_local/S7_publication_u/`. It uses the archived numerical
renderer without changing its data.

Both renderers remain in `shared_20261031/SI_Figure_S7/original_code/`:

- `export_figure_s7_panels.py`: raw algorithm time u.
- `export_figure_s7_panels_seconds.py`: stored time, already scaled as 0.175 x u.

The seconds variant is a room-temperature reference scaling, not an absolute
framework-growth calibration. Do not apply the factor twice.
The analysis convention is recorded in `../metadata/analysis_conventions.json`.

To render the stored-seconds comparison:

```bash
python 10_plot_versions/shared_20261031/SI_Figure_S7/original_code/export_figure_s7_panels_seconds.py --data-dir 07_publication_sources/SI_Figure_S7/content_checked_20261023/source_data --output-dir reproduced_local/S7_seconds --qa-dir reproduced_local/S7_seconds/qa
```

Archived generic `panel_runtime.py` files depend on their original surrounding
panel package. Use the portable commands above for the ensemble plots.
