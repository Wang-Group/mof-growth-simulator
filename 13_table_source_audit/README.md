# Literature, Entropy and Turbidity References

This folder contains literature-count sources, entropy table comparisons and
the twelve visible-turbidity observations. Original source hashes are in
`source_manifest.json`; dated table extracts retain their source-document hashes.

## Literature Counts

The adopted metal-system summary contains **69 DOI records, including 53 Zr/Hf
records**. The archived CSV also has a publication-group field: it groups
`10.1002/ange.201907074` with `10.1002/anie.201907074`, giving 68 groups
and 52 Zr/Hf groups. The published count uses DOI records.

The adopted modulator summary is **23 smaller-size outcomes among 30 studies**,
with outcome counts 23 + 2 + 2 + 1 + 2. The dated October 31 table extraction
totalling 31 belongs to an earlier version. Current adopted values are stored
in `../metadata/analysis_conventions.json`.

## Entropy

`s6_value_comparison.csv` maps translational, rotational and total entropy for
SBU, BTB, MOL_1, MOL_1', MOL_2 and MOL_3 to the source values.
All 18 dimensionless S/R values match at displayed precision
(maximum absolute difference about 4.55e-9).

The calculation source, six geometries and solvent inputs are in
`../08_entropy_inputs/source/`. BTB uses `BTBa_new.xyz`.
The calculation uses 393 K and the notebook's three-digit molecular-property
rounding. From the repository root:

```bash
python tools/reproduce_entropy.py
```

## Visible Turbidity

The twelve acetic-acid/induction-time observations are in minutes and refer to
first visible turbidity. They are separate from the in-situ DLS size trace in
group 02. Their values are retained in `selected_si_tables_20261031.json`.

## Reference Checks

```bash
python tools/audit_runtime_and_tables.py
```

This checks source hashes, counts the literature records, compares the
historical table extracts and recalculates the entropy table differences.
