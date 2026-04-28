# Explicit Zr-Ligand Growth Channel Simulations

Date: 2026-04-20

This report summarizes the simulations where we moved from an effective ligand-sequestration picture to an explicit growth channel in which prebound Zr-ligand motifs can react with the growing assembly. The main systems studied were the 2D MOL / BTB case and the UiO-66 / BDC case.

## 1. Motivation

The original KMC growth model treated external growth as adding either a free metal node or a free linker to a compatible free site. Zr concentration affected the model mostly through an effective free-metal/free-linker composition or through a lumped external addition activity.

The chemical issue we wanted to include explicitly was:

```text
Zr-AA + linker -> Zr-linker + AA
```

This prebinding reaction can have two effects.

1. It sequesters free linker, lowering the free-linker concentration available for normal growth.
2. It creates a prebound Zr-linker motif that can itself add to the growing structure.

The second point is important. If prebound Zr-linker can add freely, it may accelerate growth, because it effectively delivers both a metal node and a linker in one event. If it is geometrically hindered or changes the balance of growth and removal events, it can instead slow the time needed to reach a stable nucleus.

## 2. Model Implemented

### 2.1 Prebinding equilibrium

For both MOL and UiO-66, the prebinding state was computed before the KMC growth step.

For MOL:

- Metal node: Zr6 cluster with 12 available carboxylate-binding sites.
- Linker: BTB with 3 carboxylate arms.
- Prebound motif: Zr6-BTB.

For UiO-66:

- Metal node: Zr6 cluster with 12 available carboxylate-binding sites.
- Linker: BDC with 2 carboxylate arms.
- Prebound motif: Zr6-BDC.

The current useful scans used the `multisite_first_binding_only` model. This means the site multiplicity enters the first binding distribution, but we do not continue modeling multiple additional ligand replacements on the same cluster during growth.

### 2.2 Explicit external growth channels

The growth model now separates external events into three physical channels:

```text
free Zr6 channel      proportional to free_zr6_activity * number of linker-side free sites
free linker channel   proportional to free_linker_activity * number of metal-side free sites
prebound channel      proportional to prebound_zr_ligand_activity * compatible free sites
```

The prebound channel was the key update. It must be able to attach through both ends:

```text
prebound linker end -> existing metal site
prebound Zr6 end     -> existing linker site
```

For MOL this means Zr6-BTB can attach either through a BTB arm to a Zr site, or through the Zr6 end to a free BTB arm on the assembly. For UiO-66 this means Zr6-BDC can attach either through a BDC end to a Zr site, or through the Zr6 end to a free BDC end on the assembly.

### 2.3 Why the two-ended fix mattered

The first implementation only allowed:

```text
prebound ligand arm -> existing Zr site
```

That was incomplete. If the growing structure exposes a linker-side free site, a prebound Zr6-linker motif should be able to add by using its Zr6 end. We fixed this by:

- Splitting prebound insertion into metal-site and linker-site paths.
- Sampling prebound events from both metal free sites and linker free sites.
- Adding counters for metal-site and linker-site prebound attempts, successes, and failures.
- Preserving geometry checks, while allowing chemically intended bond contacts to be exempt from bump rejection.

Relevant implementation files:

- `MOL-Zr-Ligand/MOL_Assembly_Large_Correction_20250811.py`
- `MOL-Zr-Ligand/scan_distorted_time_to_target.py`
- `MOL-Zr-Ligand/MOL_growth_main_20250811.py`
- `UiO-66/Zr-Ligand-ExplicitChannel/UiO66_Assembly_Large_Correction_20250811.py`
- `UiO-66/Zr-Ligand-ExplicitChannel/scan_distorted_time_to_target.py`
- `UiO-66/Zr-Ligand-ExplicitChannel/UiO66_growth_main_20250811.py`

## 3. MOL / BTB Results

### 3.1 Main simulation conditions

The main retained MOL scan used:

- Target: `entity_number = 8`
- Repeats: `24`
- Max steps: `100000000`
- Linker: BTB, `8.82688 mM`
- Formic acid / capping agent: approximately `7221.027611 mM`
- Zr points: `3.38305, 6.76610, 13.53221, 27.06441, 54.12882, 108.25764, 216.51528 mM`
- Prebinding model: `multisite_first_binding_only`
- Zr6 sites: `12`
- BTB sites: `3`
- Bumping threshold: `1.8`
- Explicit prebound channel: both endpoints enabled

Retained output directory:

```text
MOL-Zr-Ligand/output/mol_active_experiment_zr_points_target8_both_endpoints_r24/scan_2026-04-16_21-20-32
```

Primary files:

- [MOL fit table](../MOL-Zr-Ligand/output/mol_active_experiment_zr_points_target8_both_endpoints_r24/scan_2026-04-16_21-20-32/fit_both_endpoints_r24_mean_kmc.csv)
- [MOL fit metadata](../MOL-Zr-Ligand/output/mol_active_experiment_zr_points_target8_both_endpoints_r24/scan_2026-04-16_21-20-32/fit_both_endpoints_r24_mean_kmc.json)
- [MOL plot](../MOL-Zr-Ligand/output/mol_active_experiment_zr_points_target8_both_endpoints_r24/scan_2026-04-16_21-20-32/both_endpoint_bugfix_effect_r24.png)
- [MOL KMC summary](../MOL-Zr-Ligand/output/mol_active_experiment_zr_points_target8_both_endpoints_r24/scan_2026-04-16_21-20-32/time_to_20_summary.csv)

### 3.2 MOL time-to-stable nucleus table

The KMC time reported here is the local simulated time to reach `entity_number = 8`, before mapping to experimental minutes.

| Zr mM | Exp min | Mean KMC s | Median KMC s | Std KMC s | Mean prebound metal successes | Mean prebound linker successes |
|---:|---:|---:|---:|---:|---:|---:|
| 3.38305 | 36 | 172.90 | 129.43 | 146.99 | 423.58 | 371.17 |
| 6.76610 | 17 | 181.75 | 102.47 | 183.75 | 438.42 | 386.21 |
| 13.53221 | 13 | 182.50 | 111.46 | 174.22 | 437.04 | 386.58 |
| 27.06441 | 11 | 152.04 | 105.44 | 151.54 | 372.75 | 316.88 |
| 54.12882 | 11 | 178.67 | 162.84 | 181.25 | 429.54 | 346.04 |
| 108.25764 | 12 | 295.89 | 280.73 | 186.39 | 677.92 | 527.75 |
| 216.51528 | 15 | 434.58 | 366.50 | 397.95 | 931.25 | 635.54 |

The two-ended prebound channel is active. Both metal-end and linker-end successes are large and nonzero at every Zr concentration. This is important because the explicit Zr6-BTB motif is not just a bookkeeping species; it is participating in growth from both compatible site types.

### 3.3 MOL mapping to experimental induction time

For MOL, the useful phenomenological mapping was:

```text
T_obs = K_seed / (free_Zr6 * free_BTB) + K_growth * T_KMC_mean
```

with `T0 = 0`.

Best fit for the current both-endpoint 24-repeat data:

```text
K_seed  = 139.0056 min*mM^2
K_growth = 0.0353458 min / simulated-second
RMSE    = 2.03 min
R2      = 0.939
```

Allowing a free intercept gave:

```text
T0      = 2.43 min
K_seed  = 131.4105 min*mM^2
K_growth = 0.0275748 min / simulated-second
RMSE    = 1.90 min
R2      = 0.947
```

### 3.4 MOL fitted decomposition

| Zr mM | Exp min | Seed waiting min | KMC-to-stable min | Pred min | Residual min |
|---:|---:|---:|---:|---:|---:|
| 3.38305 | 36 | 28.32 | 6.11 | 34.43 | 1.57 |
| 6.76610 | 17 | 14.24 | 6.42 | 20.67 | -3.67 |
| 13.53221 | 13 | 7.20 | 6.45 | 13.65 | -0.65 |
| 27.06441 | 11 | 3.69 | 5.37 | 9.06 | 1.94 |
| 54.12882 | 11 | 1.93 | 6.32 | 8.24 | 2.76 |
| 108.25764 | 12 | 1.05 | 10.46 | 11.51 | 0.49 |
| 216.51528 | 15 | 0.61 | 15.36 | 15.97 | -0.97 |

Interpretation:

- At low Zr, the dominant delay is seed waiting. The product `free_Zr6 * free_BTB` is small, so productive seed formation is slow.
- At intermediate Zr, seed waiting collapses and the system reaches the fastest induction region.
- At high Zr, the local KMC-to-stable term rises because the free BTB pool is reduced and prebound/growth/removal balance becomes less favorable. This reproduces the experimental upturn at high Zr.

This was the most chemically coherent result for the MOL data. The simulation alone gives the local time for a seed to become stable; the extra seed-waiting term accounts for the probability of starting such a productive seed in the experimental volume.

## 4. UiO-66 / BDC Results

### 4.1 Coordinate KMC diagnostic before the high-AA fit

There was an earlier coordinate KMC scan for UiO-66:

- Target: `entity_number = 8`
- Repeats: `24`
- Zr points: `24, 32, 40, 48, 64 mM`
- Prebinding model: `multisite_first_binding_only`
- This run was from before the final double-endpoint UiO-66 patch, but it is still useful as a diagnostic for how strongly the local KMC time rises with Zr.

Retained directory:

```text
UiO-66/Zr-Ligand-ExplicitChannel/output/explicit_channel_time8_target_scan_repeats24/scan_2026-04-15_14-19-56
```

Summary:

| Zr mM | Mean KMC s | Median KMC s | Std KMC s | Mean prebound fraction |
|---:|---:|---:|---:|---:|
| 24 | 78.97 | 52.85 | 80.33 | 0.321 |
| 32 | 107.19 | 66.16 | 111.33 | 0.387 |
| 40 | 214.86 | 114.04 | 274.70 | 0.442 |
| 48 | 616.99 | 408.64 | 626.02 | 0.487 |
| 64 | 1108.22 | 957.09 | 917.53 | 0.559 |

This already showed a strong high-Zr slowdown in the local KMC dynamics. The slowdown appears when prebound fraction becomes large and free BDC is suppressed.

### 4.2 UiO-66 double-endpoint fix

The same two-ended prebound correction was applied to UiO-66:

```text
Zr6-BDC ligand end -> existing Zr site
Zr6-BDC Zr6 end    -> existing BDC site
```

A direct smoke test showed that linker-end insertion succeeds when `BUMPING_THRESHOLD = 1.8`. With `BUMPING_THRESHOLD = 2.0`, the linker-end insertion was essentially rejected by geometry checks. This mirrors the MOL case, where a threshold around `1.8` was needed to avoid rejecting chemically intended contacts.

### 4.3 UiO-66 experimental fit conditions

The UiO-66 experimental points were taken from:

```text
C:/Users/yibinjiang/Documents/xwechat_files/wxid_w2m1w39nju0q22_a138/msg/file/2026-04/MOL-UiO66-induction.csv
```

For the UiO-66 left-side series:

- BDC linker concentration: `4 mM`
- AA concentration: `2400 mM`
- ZrCl4 points: `2, 4, 8, 16, 32 mM`
- Experimental induction times: `82, 114, 200, 223, 330 min`

A full coordinate KMC run under these high-AA conditions was attempted, but it timed out: reaching `entity_number = 8` became a rare first-passage event. Short diagnostics showed many successful prebound insertions, but AA/formate removal repeatedly broke the cluster back to small entity numbers.

For this reason, the high-AA UiO-66 fit used a graph-level first-passage KMC approximation:

- Keeps Zr6/BDC site counts: `12` and `2`.
- Keeps free-Zr6, free-BDC, and prebound Zr6-BDC channels.
- Keeps AA/formate bond-removal pressure.
- Omits coordinate bump checks.

This is reasonable as a diagnostic because the short coordinate runs indicated that geometry was not the limiting factor after the `1.8` bump threshold correction; the limiting factor was the growth-removal first-passage balance.

Retained output directory:

```text
UiO-66/Zr-Ligand-ExplicitChannel/output/uio66_exp_L4_AA2400_fit_graph_kmc
```

Primary files:

- [UiO-66 fit table](../UiO-66/Zr-Ligand-ExplicitChannel/output/uio66_exp_L4_AA2400_fit_graph_kmc/uio66_L4_AA2400_fit_table.csv)
- [UiO-66 fit metadata](../UiO-66/Zr-Ligand-ExplicitChannel/output/uio66_exp_L4_AA2400_fit_graph_kmc/uio66_L4_AA2400_fit.json)
- [UiO-66 fit plot](../UiO-66/Zr-Ligand-ExplicitChannel/output/uio66_exp_L4_AA2400_fit_graph_kmc/uio66_L4_AA2400_fit.png)

### 4.4 UiO-66 fit results

The tested model was:

```text
T_obs = T0 + K_seed / (free_Zr6 * free_BDC) + K_growth * T_KMC_graph_to_entity8
```

Unconstrained fit:

```text
T0       = 28.80 min
K_seed   = -139.15 min*mM^2
K_growth = 0.003549 min / KMC-second
RMSE     = 14.23 min
R2       = 0.974
```

Physical nonnegative fit:

```text
T0       = 0 min
K_seed   = 0 min*mM^2
K_growth = 0.003461 min / KMC-second
RMSE     = 40.23 min
R2       = 0.788
```

Table:

| Zr mM | Exp min | Free Zr6 mM | Free BDC mM | Prebound fraction | Graph KMC-to-8 s | Unconstrained pred min | Physical pred min |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 82 | 0.331 | 3.973 | 0.0068 | 43008 | 75.66 | 148.84 |
| 4 | 114 | 0.662 | 3.946 | 0.0135 | 45182 | 135.91 | 156.37 |
| 8 | 200 | 1.324 | 3.893 | 0.0267 | 49810 | 178.61 | 172.39 |
| 16 | 223 | 2.649 | 3.792 | 0.0520 | 60280 | 228.91 | 208.62 |
| 32 | 330 | 5.300 | 3.605 | 0.0988 | 86882 | 329.90 | 300.69 |

Interpretation:

- Unlike the MOL/Hf-BTB data, the UiO-66 series does not support a positive seed-waiting term of the form `K_seed/(free_Zr6*free_BDC)`.
- The unconstrained fit gives a negative `K_seed`, which is not physically meaningful in the seed-waiting picture.
- When constrained to physical nonnegative coefficients, `K_seed` becomes zero.
- The experimental trend is therefore mostly explained by the KMC growth/removal slowdown as Zr increases, not by low-Zr seed waiting.

Chemically, this makes sense for this specific UiO-66 dataset because the low-Zr points are not slow. The data go from `82 min` at `2 mM` Zr to `330 min` at `32 mM` Zr. A positive low-Zr seed-waiting penalty would push the `2 mM` point upward, which is opposite to the observed trend.

## 5. Cross-System Interpretation

### 5.1 MOL has two competing timescales

For MOL, the experimental curve is nonmonotonic:

- Low Zr is slow.
- Intermediate Zr is fastest.
- High Zr becomes slower again.

The two-term model captures this:

```text
T_obs = seed waiting + local KMC-to-stable growth
```

Low Zr:

```text
seed waiting dominates
```

High Zr:

```text
prebinding/linker depletion/local growth-removal dynamics dominate
```

This is why MOL can be explained by adding an upper-layer seed-waiting term on top of the local KMC simulation.

### 5.2 UiO-66 in the high-AA dataset is different

For the UiO-66 dataset with `L = 4 mM` and `AA = 2400 mM`, the observed induction time increases with Zr over the measured range. There is no low-Zr slow branch in the provided points.

Therefore:

```text
positive seed waiting is not required and is actually disfavored
```

The more useful explanation is:

```text
increasing Zr increases prebound Zr-BDC / AA-controlled removal pressure and slows local first passage to a stable nucleus
```

This does not mean seed waiting is never relevant for UiO-66. It means this particular experimental window does not show a low-Zr induction-time penalty that would require the seed-waiting term.

## 6. Important Caveats

1. The coordinate KMC time is a local first-passage time after placing an initial Zr6 seed. It is not automatically the experimental induction time.

2. Experimental induction time may include an upper-layer waiting time for productive seed activation in a volume or voxel:

```text
T_obs = T_seed_waiting + T_local_growth_to_stable
```

3. The MOL data support a positive `T_seed_waiting` term. The UiO-66 high-AA data do not.

4. Full coordinate KMC under high-AA UiO-66 conditions was too slow to sample directly because entity number frequently collapsed before reaching 8. The graph-level first-passage model is a diagnostic approximation, not a replacement for the geometry model.

5. The bumping threshold matters. For prebound motif insertion, `BUMPING_THRESHOLD = 2.0` rejected many chemically intended insertions, while `1.8` allowed the expected channel. This threshold should be treated as a calibrated geometric tolerance, not a universal constant.

6. The current model includes the first prebinding event and explicit motif addition, but it does not model all possible subsequent ligand substitutions on the same Zr cluster during growth.

## 7. Working Conclusions

1. Explicit Zr-ligand motifs should be kept as a separate growth channel. Treating them only as ligand sequestration misses a real pathway.

2. The prebound channel must be two-ended. A prebound Zr6-linker motif can attach through the linker end to a metal site or through the Zr6 end to a linker site.

3. For MOL / BTB, the model now gives a coherent explanation of the experimental nonmonotonic Zr dependence:

```text
low Zr: seed waiting slow
intermediate Zr: fastest
high Zr: local KMC growth to stable nucleus slows
```

4. For UiO-66 / BDC at `L = 4 mM`, `AA = 2400 mM`, the data mainly support a high-Zr local growth/removal slowdown. A positive seed-waiting term is not supported by the available points.

5. The most useful next step is not to blindly run more long coordinate KMC trajectories. Instead, we should use short coordinate runs to calibrate graph-level or reduced first-passage models, then reserve full coordinate KMC for selected validation points.

## 8. Suggested Next Simulations

1. For MOL, repeat the active Zr scan at `target_entities = 12` or `20` for fewer selected Zr points to see whether the fitted `K_growth` remains stable.

2. For UiO-66, run a smaller coordinate validation set at `Zr = 2, 8, 32 mM`, but with adaptive stopping and trajectory diagnostics, not brute-force 24 repeats.

3. Add output diagnostics for collapse events:

```text
number of additions
number of removals
maximum entity number reached
time spent at entity number 1-3
first time reaching entity number 4, 6, 8
```

4. Test whether the UiO-66 high-Zr slowdown is more sensitive to AA concentration, prebinding constant, or bump threshold.

5. Keep the two-layer interpretation explicit in future reports:

```text
experimental induction time = seed/activation waiting + local KMC stabilization time
```

but fit the seed/activation term only when the experimental data require it.

