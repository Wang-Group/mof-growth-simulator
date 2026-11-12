# MOF/MOL Growth: Code and Analysis Data

Event-based kinetic Monte Carlo (KMC) simulation codes and analysis tools for
molecular assembly, nucleation, growth, layer morphology and defect repair in
metal-organic frameworks and layers.

The repository includes simulation engines, molecular models, seed structures,
parameter files, compact analysis datasets, experimental measurements, and
plotting and fitting scripts. Browse the scientific topics below or the
[code entry points](docs/CODE_INDEX.md).

## Contents

| Scientific topic | Folders | Contents |
|---|---|---|
| UiO-66 growth trajectories | [01](01_uio66_long_growth/) | Analysis of 24 repeated trajectories, first-passage times, residence times and last-saved entity-13 structures |
| Experimental DLS | [02](02_dls/) | Size measurements, apparent-progress processing, Gualtieri fitting and plots |
| BTB growth and motifs | [03](03_btb_trajectory/), [04](04_btb_motifs/), [15](15_fig5d_trajectory_lineage/) | Growth curves, motif frequencies, classification code, reference structures and row-to-run mappings |
| HKUST-1 and PCN-240 growth | [18](18_hkust1_pcn240/) | Framework-specific simulation engines, growth runners and molecular model inputs |
| Mixed-node growth and vacancy repair | [05](05_mixed_zr6_zr12/), [06](06_vacancy_repair/), [11](11_s21_april03/) | Zr6/Zr12 evolution, defect-filling curves and representative structures |
| Nucleation and seeded/surface growth | [07](07_publication_sources/), [09](09_recovered_runtime/), [12](12_runtime_provenance/) | Simulation engines, molecular models, parameter grids, seed inputs and endpoint statistics |
| Controlled modulator geometry | [source data](07_publication_sources/SI_tables_sources_20261023/Table_S20/matched_source_files/) | Twelve identities, 27,648 terminal records and matched assembly/growth comparisons |
| Ring-closure entropy | [08](08_entropy_inputs/) | Calculation code, molecular geometries and entropy results for six species |
| BTB kinetic boundary and sensitivity | [16](16_fig5c_reconstruction/) | Success probabilities, boundary plots, variance indices and bootstrap confidence intervals |
| Modulator identity and ligand exchange | [17](17_submission_sources/) | Identity comparisons, NMR measurements and acquisition mappings |
| Finite-cluster structure and scattering | [07](07_publication_sources/), [17](17_submission_sources/figure6c/) | PXRD analyses and matched-size BTB comparisons |
| Source and plot records | [10](10_plot_versions/), [13](13_table_source_audit/), [14](14_fig5_source_followup/) | Plotting versions and source correspondence |

## Quick Start

Use Python 3.11 or newer with the packages in [requirements.txt](requirements.txt).
Install Arial to reproduce the figure typography.

```bash
python -m pip install -r requirements.txt
python reproduce.py --output-dir reproduced_local
```

This generates the trajectory, DLS, motif-frequency, mixed-node growth and
vacancy-repair plots from the included analysis data. New outputs are written to
`reproduced_local/`; reference outputs are in `reproduced/`.

Additional analysis entries:

```bash
python 10_plot_versions/render_s7.py
python tools/reproduce_entropy.py
python tools/reproduce_s20.py
python 16_fig5c_reconstruction/render.py
python tools/reproduce_btb_sensitivity.py
python 11_s21_april03/plot_saved_trace.py --output-dir reproduced_local/mixed_seed_april03
```

The [code index](docs/CODE_INDEX.md) identifies the simulation and analysis
sources. The [modulator-geometry guide](docs/S20_REPRODUCTION.md) describes the
calculation of matched statistics from the terminal records. Dataset-specific
time coordinates, structure definitions and processing methods are documented
in the corresponding folders.

## Checks

```bash
python verify_package.py --output-dir reproduced_local
python verify_publication.py --report reproduced_local/publication_check.json
python -m unittest discover -s tests -v
```

Source hashes are recorded in [CHECKSUMS.sha256](CHECKSUMS.sha256).
Included inputs and external simulation archives are described in
[data scope](docs/DATA_SCOPE.md); [data conventions](docs/PROVENANCE_NOTES.md)
define time coordinates and structure-selection methods.

## License and Citation

Original code is licensed under MIT and project-produced data under CC BY 4.0.
See [LICENSING.md](LICENSING.md) for attribution and third-party notices.
Cite this repository and the release tag or commit used for the analysis.
