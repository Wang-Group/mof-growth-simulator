# UiO-66 In Situ DLS Gualtieri Fit

This folder is the self-contained source package for the UiO-66 in situ DLS
analysis described in the manuscript and Supporting Information.

## Contents

- `data/260316.csv`: original DLS export. The first time column excludes the
  90 s preparation period; the third column includes it. The fit uses the last
  two columns.
- `fit_uio66_dls_gualtieri.py`: data processing, Gualtieri fitting, CSV export,
  and PNG/SVG rendering.
- `output/uio66_dls_gualtieri_processed_trace.csv`: raw size, smoothed size,
  apparent progress, and fitted values at each measured time.
- `output/uio66_dls_gualtieri_fit_curve.csv`: dense fitted curve.
- `output/uio66_dls_gualtieri_parameters.csv`: fitted parameters and fit
  quality.
- `output/uio66_dls_gualtieri_fit.png` and `.svg`: manuscript-style overlay.
- `output/uio66_dls_gualtieri_summary.md`: human-readable result summary.

## Processing and Model

The hydrodynamic-size trace is processed as follows:

1. Apply a centered three-point rolling median.
2. Define the baseline as the median of the first five smoothed points.
3. Define the plateau as the mean of the last five smoothed points.
4. Normalize the smoothed trace between the baseline and plateau.
5. Apply a cumulative maximum to obtain a monotonic apparent-progress proxy.

The fitted equation is

```text
alpha(t) = [1 / (1 + exp(-(t - aN) / bN))]
           * [1 - exp(-(kG * t)^nG)]
```

The apparent-progress proxy is a macroscopic kinetic descriptor. It should not
be interpreted as direct crystallinity or directly equated with KMC entity
number.

## Reproduce

From the repository root, run:

```powershell
conda run -n base python .\UiO-66\DLS_Gualtieri\fit_uio66_dls_gualtieri.py
```

The deterministic fit uses random seed `3001` and should reproduce:

- `nG = 2`
- `aN = 215.538827 s`
- `bN = 2.512205 s`
- `kG = 1.92799484e-3 s^-1`
- `R^2 = 0.972658`
- `RMSE = 0.051750`

The reported `R^2` is calculated against the processed monotonic
apparent-progress proxy.

## Provenance

The archived raw CSV has SHA-256:

```text
2364e652ad4bcba27f786b6e7ca9d11ef5cf55a59d2e7036079d57e599b14422
```

This matches the original file received as `260316.csv`.
