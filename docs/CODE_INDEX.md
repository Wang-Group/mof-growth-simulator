# Code Entry Points

Run the commands below from the repository root. Source folders retain their
original names; this index organizes them by scientific topic.

## Simulation Sources

| Topic | Location | Contents |
|---|---|---|
| UiO-66 repeated growth | `01_uio66_long_growth/source/` | Simulation source and run conditions |
| BTB growth | `03_btb_trajectory/source/` | Executed growth runner and assembly engine |
| HKUST-1 and PCN-240 growth | [18_hkust1_pcn240](../18_hkust1_pcn240/) | Original runners, framework-specific engines, molecular models and setup instructions |
| Mixed-node growth and vacancy repair | `05_mixed_zr6_zr12/original/`, `06_vacancy_repair/original/`, `11_s21_april03/source_archive/` | Growth/repair scripts and initialization variants |
| Nucleation, seeded growth, surface growth and finite baths | `07_publication_sources/SI_tables_sources_20261023/Table_S28/code/` | Campaign runners, selectors and analysis scripts |
| Molecular engines and models | `09_recovered_runtime/` | Executed engines and molecular models |
| Campaign inputs | `12_runtime_provenance/` | Parameter manifests, small seed/host inputs and source-version mappings |

Production runners use their corresponding model, parameter and seed paths.
Consult the folder README and run manifest when configuring a new calculation;
continuation runs also take the checkpoint specified by that runner.

## Analysis and Plotting

| Topic | Entry | Output or operation |
|---|---|---|
| Combined plotting | `python reproduce.py --output-dir reproduced_local` | Plots for groups 01-06 from included analysis data |
| UiO-66 trajectory statistics | `01_uio66_long_growth/scripts/` | First-passage, residence-time and structure analyses |
| UiO-66 algorithm-time plots | `python 10_plot_versions/render_s7.py` | First-passage and residence-time panels in raw algorithm time u |
| DLS/Gualtieri fit | `02_dls/fit_uio66_dls_gualtieri.py` | Apparent-progress fit, parameters and diagnostic plots |
| BTB motif frequencies | `reproduce.py`, `04_btb_motifs/` | Frequencies and representative structures |
| BTB motif classification | `15_fig5d_trajectory_lineage/data/metadata.zip` | Classifier/extractor notebook sources and per-run code |
| BTB motif record checks | `python -m tools.audit_fig5d_lineage` | Row/index checks and classification of four packaged examples; use `--full-source-archive PATH` for the full cohort |
| BTB kinetic boundary | `python 16_fig5c_reconstruction/render.py` | Interpolated P=0.5 isosurface and boundary plots |
| BTB sensitivity | `python tools/reproduce_btb_sensitivity.py` | First-order/total-effect indices and 2,000-resample bootstrap confidence intervals |
| Ring-closure entropy | `python tools/reproduce_entropy.py` | Entropy calculations for six species |
| Controlled modulator geometry | `python tools/reproduce_s20.py` | Statistics for 864 groups and nine matched comparisons; [method](S20_REPRODUCTION.md) |
| April 3 mixed-node growth | `python 11_s21_april03/plot_saved_trace.py --help` | Plotting of the saved April 3 dataset |
| Exchange and modulator-identity data | `python -m tools.verify_submission_sources` | Source checks and identity regressions |

The April 3 and April 10 mixed-seed datasets have separate plotting entries.
The combined renderer uses the April 10 dataset. Notebook source code is provided
inside the ZIP alongside its original per-run code and indices.

## Verification

```bash
python verify_package.py
python verify_publication.py
python -m unittest discover -s tests -v
```

Machine-readable reports record the individual checks. Source versions and
file hashes are retained in the group manifests and `metadata/`.
