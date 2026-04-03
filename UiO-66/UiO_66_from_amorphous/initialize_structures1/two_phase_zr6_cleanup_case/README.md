# Canonical Two-Phase Zr12 Cleanup Case

This folder documents the retained mixed-nuclei cleanup chain that starts from a mixed Zr6/Zr12 amorphous seed and ends at a pure-Zr6 result.

## Retained Data

The kept output chain is:

1. `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\output\mixed_nuclei\two_phase_amorphous_equilibrate_zr6only\two_phase_eqbond_zr6only_default`
2. `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\output\mixed_nuclei\two_phase_amorphous_equilibrate_zr6only\two_phase_eqbond_zr6only_continue_from10`
3. `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\output\mixed_nuclei\two_phase_amorphous_equilibrate_zr6only\two_phase_eqbond_zr6only_from9_recheck`
4. `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\output\mixed_nuclei\two_phase_amorphous_equilibrate_zr6only\two_phase_eqbond_zr6only_continue_from8`
5. `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\output\mixed_nuclei\two_phase_amorphous_equilibrate_zr6only\two_phase_eqbond_zr6only_long_from6`
6. `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\output\mixed_nuclei\two_phase_amorphous_equilibrate_zr6only\two_phase_eqbond_zr6only_long_from2`

These 6 directories are the canonical chain from `Zr12 = 36` to `Zr12 = 0`.

## Canonical Input and Final Output

- Earliest seed:
  `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\output\mixed_nuclei\two_phase_amorphous_equilibrate_zr6only\two_phase_eqbond_zr6only_default\two_phase_eqbond_zr6only_default__phase1_seed.json`
- Final pure-Zr6 result:
  `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\output\mixed_nuclei\two_phase_amorphous_equilibrate_zr6only\two_phase_eqbond_zr6only_long_from2\two_phase_eqbond_zr6only_long_from2__stage02__rep03__seed103002.json`

Key counts for this canonical pair:

- Phase-1 seed: `Zr6 = 26`, `Zr12 = 36`, `BDC = 138`, `total_entities = 200`
- Final pure-Zr6 result: `Zr6 = 2616`, `Zr12 = 0`, `BDC = 7497`, `total_entities = 10113`
- Net change: `delta_Zr6 = +2590`, `delta_Zr12 = -36`, `delta_BDC = +7359`
- Canonical Zr12 cleanup sequence:
  `36 -> 23 -> 16 -> 12 -> 10 -> 9 -> 9 -> 8 -> 7 -> 6 -> 6 -> 4 -> 3 -> 2 -> 1 -> 0`

Shape change:

- `principal_axis_ratio_1_3`: `4.737869 -> 1.334663`
- `span_ratio_max_min`: `1.127142 -> 1.050810`

This means the retained final structure is both pure Zr6 and much more isotropic than the initial mixed amorphous seed.

## Runner

Use the runner in this folder to regenerate the side-by-side quantitative comparison from the retained canonical data:

```powershell
$env:PYTHONIOENCODING='utf-8'
& "C:\Users\yibinjiang\AppData\Local\Programs\Python\Python311\python.exe" `
  "C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\two_phase_zr6_cleanup_case\runner.py"
```

The runner writes:

- `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\output\mixed_nuclei\two_phase_amorphous_equilibrate_zr6only\canonical_report\canonical_seed_vs_final.json`
- `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\output\mixed_nuclei\two_phase_amorphous_equilibrate_zr6only\canonical_report\canonical_seed_vs_final.md`

## Constant-Size Control Chain

To test a strict constant-size phase-2 limit, use:

```powershell
$env:PYTHONIOENCODING='utf-8'
& "C:\Users\yibinjiang\AppData\Local\Programs\Python\Python311\python.exe" `
  "C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\two_phase_zr6_cleanup_case\runner_constant_size.py"
```

This writes:

- `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\output\mixed_nuclei\two_phase_constant_size_closed_anneal\constant_size_closed_phase1seed\constant_size_closed_phase1seed.summary.json`

Current constant-size result from the same phase-1 seed:

- entity counts stay fixed:
  `Zr6 = 26`, `Zr12 = 36`, `BDC = 138`, `total_entities = 200`
- linked pairs relax from `216` to about `103.5` on average across 4 replicates
- ready pairs rise from `0` to about `112.5`
- Zr12 mean coordination drops from `3.8333` to about `1.8194`
- the single connected cluster does not stay intact; it fragments strongly:
  average final `component_count = 97.0`, average largest component size `= 31.25`

Important interpretation:

- This constant-size control is a closed bond-network anneal on a frozen geometry.
- It disables external addition and disables entity deletion.
- It therefore tests reversible internal linking only.
- Under the current model, that control does not convert Zr12 into Zr6 and does not produce a larger single crystal.
- Instead it relaxes toward a much less connected, fragmented mixed network.

## Size-Stabilized Zr6-Only Follow-Up

To keep the mixed-seed follow-up near constant size while still allowing:

- external `Zr6` and `BDC` addition
- reversible bond breaking

use:

```powershell
$env:PYTHONIOENCODING='utf-8'
& "C:\Users\yibinjiang\AppData\Local\Programs\Python\Python311\python.exe" `
  "C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\two_phase_zr6_cleanup_case\runner_size_stabilized.py"
```

The retained size-stabilized run is:

- `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\output\mixed_nuclei\two_phase_size_stabilized_zr6only\size_stabilized_phase1seed_target210_pruned`

Current default control parameters in `runner_size_stabilized.py`:

- `target_total_entities = 210`
- `size_feedback_gain = 20.0`
- `control_update_interval = 10`
- `size_deadband_fraction = 0.0`
- `max_formate_ratio = 200.0`

This retained version also deletes any detached fragment as a whole after bond removal, keeping only the largest connected component in the assembly. Because that cleanup adds an extra shrinkage channel, the size target is biased upward to `210` to keep the realized mean trajectory size near the original `200`-entity seed.

Aggregate result across 4 replicates from the same phase-1 seed:

- start seed: `Zr6 = 26`, `Zr12 = 36`, `BDC = 138`, `total_entities = 200`
- mean trajectory size: `199.6628 +- 0.3240`
- mean final size: `209.5`
- mean net size drift: `+9.5`
- mean final composition: `Zr6 = 66.25`, `Zr12 = 0.0`, `BDC = 143.25`
- mean net composition change: `delta_Zr6 = +40.25`, `delta_Zr12 = -36.0`, `delta_BDC = +5.25`
- detached-fragment cleanup activity: about `3106.25` prune events and `48003.25` deleted entities per replicate
- connectivity after rebuilding references from the saved PKLs: all 4 final replicates remain single connected clusters

Representative retained outputs:

- summary:
  `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\output\mixed_nuclei\two_phase_size_stabilized_zr6only\size_stabilized_phase1seed_target210_pruned\size_stabilized_phase1seed_target210_pruned.summary.json`
- representative JSON:
  `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\output\mixed_nuclei\two_phase_size_stabilized_zr6only\size_stabilized_phase1seed_target210_pruned\size_stabilized_phase1seed_target210_pruned__rep04__seed105003.json`
- representative PKL:
  `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\output\mixed_nuclei\two_phase_size_stabilized_zr6only\size_stabilized_phase1seed_target210_pruned\size_stabilized_phase1seed_target210_pruned__rep04__seed105003.pkl`

## Cleanup Policy

All other exploratory `mixed_nuclei` output directories are treated as disposable waste data and can be removed.

The only kept mixed-nuclei outputs are:

- the 6 canonical chain directories listed above
- the generated `canonical_report` directory
- the retained size-stabilized directory:
  `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\output\mixed_nuclei\two_phase_size_stabilized_zr6only\size_stabilized_phase1seed_target210_pruned`

## Canonical Chain With Detached-Fragment Pruning

To replay the retained canonical best-path chain with a new rule that deletes any detached fragment as a whole after bond removal, use:

```powershell
$env:PYTHONIOENCODING='utf-8'
& "C:\Users\yibinjiang\AppData\Local\Programs\Python\Python311\python.exe" `
  "C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\two_phase_zr6_cleanup_case\runner_canonical_pruned_chain.py"
```

Retained output:

- `C:\Users\yibinjiang\Documents\GitHub\mof-growth-simulator\UiO-66\UiO_66_from_amorphous\initialize_structures1\output\mixed_nuclei\two_phase_amorphous_equilibrate_zr6only_pruned\canonical_pruned_chain`

Current result:

- the original canonical large-growth behavior is not recovered
- `Zr12` is cleared immediately in `default_stage01`
- the chain stays as a single connected cluster throughout, but it collapses to a small object instead of growing
- best path after `default_stage01`: `total_entities = 61`, `Zr6 = 20`, `Zr12 = 0`
- largest best-path size reached later in the replayed plan: `total_entities = 134` at `long2_stage01`
- final canonical-pruned best path at `long2_stage02`: `total_entities = 84`, `Zr6 = 26`, `Zr12 = 0`

Interpretation:

- adding detached-fragment deletion fixes the topological artifact that previously allowed many disconnected clusters to coexist and grow
- under the same equal-make/break chemistry, once detached fragments are actually removed from the system, the canonical chain no longer supports runaway growth to a `10^4`-entity pure-`Zr6` aggregate
- this means the previous `~10113`-entity result depended on retaining detached fragments inside the simulation box
