# BTB Structural Analysis Sources

This folder retains two historical analysis sources:

- `source/unique_structures.ipynb`: original classification source cells, with execution outputs removed.
- `source/make_entity_structure_grid.py`: the original structure-grid plotting script.

The notebook compares SBU/linker counts and element-wise radial-histogram
similarity greater than 0.8. This is a geometric similarity classifier, not an
exact graph-isomorphism test. Configure its input and output directories before use.

The complete row/run index, classification records and reproducible classifier
examples are in [group 15](../15_fig5d_trajectory_lineage/README.md).
The boundary and sensitivity analysis code is in
[group 16](../16_fig5c_reconstruction/README.md).

`source_manifest.json` preserves the code hashes and original archive references.
Bulk structures and trajectory pickles remain external under the repository's
[data scope](../docs/DATA_SCOPE.md).

From the repository root, verify these source files with:

```bash
python -m tools.audit_fig5_sources
```
