# BTB high-rho pilot: N=1000

Approved 7 September 2026: four removal coefficients (10, 30, 50, 100), eight
independent stochastic trajectories per condition, all starting from the same
reviewed 30-entity molecular seed. This is conditional seeded growth, not a
measurement of homogeneous nucleation or a sample of different initial seeds.

## Fixed inputs and execution

- Total target N=1000 (metal clusters plus BTB molecules).
- s_close=1, atomic metal/BTB=1, node fraction=1/7, external activity=1.
- Pure hexanuclear-cluster feed; both reviewed BTB orientations.
- Unrestricted 3D molecular alignment and collision testing, all 12 cluster sites.
- Every existing linkage has the same removal coefficient. Smaller disconnected
  fragments are removed by the reviewed engine. No planar constraint or new
  coordination-accessibility rule is introduced.
- The seed's preparation parameters appear in its filename; production uses the
  inputs above. `seed_retention_threshold=0.8` is only a recorded diagnostic:
  it neither protects the seed nor terminates a trajectory.
- Fixed reservoir inputs, without DMF decomposition or reservoir depletion.
- Save molecular structures and geometry diagnostics at N=250, 500, 1000, and at
  termination. Stop at N=1000, 100 million events, or four simulation-wall hours.
  Final structure export/analysis can add modest time beyond the wall cap.
- 16 workers, each numerical runtime limited to one CPU thread, on host 5090.
- Fresh SHA-derived random seeds are listed in manifest.csv. Existing trajectories
  are controls for method consistency, not additional independent replicates.

The five production Python modules, geometry engine, four molecular files and
seed are copied to a campaign-specific `snapshot/` on 5090 and verified before
launch. No simulation dynamics are edited. The baseline N=80 first-passage replay
checks the snapshot against a completed N=1000 trajectory before production.
The remote /usr/bin/python3 and its existing NumPy geometry adapters are retained
to match the preceding runs; no environment is installed or upgraded.

## Analysis defined before examining the pilot

1. Report all 8 outcomes per condition: target reached, finite-cap censoring,
   no enabled event, or software error. Report initial-seed loss separately; it
   is not automatically an absorbing chemical loss channel.
2. Compare morphology at the SAME total N, and also report metal-cluster count
   and composition. Use cluster-centre PCA thickness, lateral extents and
   eigenvalue ratios; inspect coordinates for out-of-plane branches. Do not
   equate archived transitive layer-domain counts with literal thin-layer counts.
3. Plot growth histories and first-passage model times. Give the target-reached
   fraction separately from conditional-on-reaching size/shape summaries. Wall
   caps are computational limits, not identical physical observation times.
4. Calculate finite-cluster atomic Debye PXRD at N=250/500/1000 with identical
   wavelength, form factors, angular grid and smoothing to the preceding series.
   Compare with a documented kgd reference; the three main hk0 sticks are not a
   complete diffraction standard. Save numeric spectra, not only images.
5. Track the additional feature near 5.8 degrees using the established 5.55-6.10
   degree window and a fixed detection rule across all conditions. Report absent
   resolved maxima explicitly, without replacing them by zeros or assigning a
   unique phase. Check node-centre scattering and morphology before attributing
   peak loss to suppression of out-of-plane growth: finite size can obscure peaks.
6. The fixed rho_off coefficient is not the time-dependent total removal/addition
   propensity ratio. Retain sampled site/bond counts so this distinction remains
   recoverable. NMR conversion fields remain metadata, not validated productive
   growth times in seconds.

Expansion to 32 trajectories per condition is a later stage, conditional on pilot
cost and interpretability; it is not launched automatically. No cap increases,
replacement of failed trajectories, manuscript changes or student-ZIP overwrites.

## Locations

Remote: `/home/wave/MOL_growth_recalculation_20260830/btb_high_rho_N1000_20260907_v1`

- `manifest.csv`, `design.json`: exact planned parameters and input SHA-256 values.
- `snapshot/`: actual unmodified executable code, molecular files and seed.
- `preflight/`: smoke-test output and verification record (excluded from production).
- `results/`: terminal JSONs, including capped outcomes; `results/structures/`: MOL2.
- `launch.json`, `campaign.log`, `completion.json`: execution records.

Local directory has the same campaign name below `analysis_workspace/rewrite/recalculation_campaign_20260830`.

Inspect without launching: `python3 remote_control.py status` on 5090.
`remote_control.py launch` has a one-time lock and requires a successful preflight.
