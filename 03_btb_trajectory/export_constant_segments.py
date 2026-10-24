"""Optional export from the trusted original pickle; no simulation execution.

The original pickle is intentionally not distributed. Only unpickle a trusted
simulation file: Python pickle can execute code. This script is not needed for
reproduce.py, which reads the already exported CSV.gz.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_pickle", type=Path)
    parser.add_argument("output_csv_gz", type=Path)
    args = parser.parse_args()
    if args.output_csv_gz.exists():
        raise FileExistsError(args.output_csv_gz)
    with args.source_pickle.open("rb") as handle:
        arr = np.asarray(pickle.load(handle), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or np.any(np.diff(arr[:, 0]) < 0):
        raise ValueError("Expected monotonic two-column [time, entity] trajectory")
    starts = np.r_[0, np.flatnonzero(np.diff(arr[:, 1]) != 0) + 1]
    ends = np.r_[starts[1:] - 1, len(arr) - 1]
    data = pd.DataFrame(dict(row_start=starts, row_end_inclusive=ends,
                             time_start_stored=arr[starts, 0], time_end_stored=arr[ends, 0],
                             entity=arr[starts, 1].astype(int)))
    assert np.array_equal(np.repeat(data.entity.to_numpy(), ends - starts + 1), arr[:, 1])
    args.output_csv_gz.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_csv_gz, index=False, float_format="%.17g",
                compression={"method": "gzip", "mtime": 0})
    restored = pd.read_csv(args.output_csv_gz, float_precision="round_trip")
    assert np.array_equal(restored.time_start_stored.to_numpy(), arr[starts, 0])
    assert np.array_equal(restored.time_end_stored.to_numpy(), arr[ends, 0])
    with args.source_pickle.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    print(json.dumps(dict(source_sha256=checksum, source_rows=len(arr),
                          exported_segments=len(starts), time_conversion="none"), indent=2))


if __name__ == "__main__":
    main()
