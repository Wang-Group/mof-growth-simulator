#!/usr/bin/env python3
"""Memory-bounded finite-cluster Debye PXRD for N=10000 aggregates.

The calculation is mathematically identical to the validated histogram-based
Debye implementation used for N=1000.  Pair distances are accumulated over
coordinate blocks so that no O(N^2) distance vector is materialized.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d


HERE = Path(__file__).resolve().parent
FIGURE3 = HERE.parent
sys.path.insert(0, str(FIGURE3))
from compute_finite_cluster_pxrd import (  # noqa: E402
    Cluster,
    atomic_form_factor,
    parse_mol2,
    periodic_reference_peaks,
)


def chunked_pair_distance_histograms(
    cluster: Cluster,
    *,
    device: torch.device,
    bin_width: float,
    block_size: int,
) -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray], dict[str, int], dict[str, object]]:
    coords = torch.as_tensor(cluster.coordinates, dtype=torch.float32, device=device)
    diagonal = torch.linalg.vector_norm(coords.max(dim=0).values - coords.min(dim=0).values)
    max_distance = float(diagonal.detach().cpu()) + bin_width
    bins = max(2, int(math.ceil(max_distance / bin_width)))
    upper = bins * bin_width
    centers = (np.arange(bins, dtype=np.float32) + 0.5) * bin_width

    elements_np = np.asarray(cluster.elements)
    element_arrays: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    for symbol in sorted(set(cluster.elements)):
        indices = np.flatnonzero(elements_np == symbol)
        element_arrays[symbol] = coords[torch.as_tensor(indices, device=device)]
        counts[symbol] = int(len(indices))

    histograms: dict[tuple[str, str], np.ndarray] = {}
    pair_counts: dict[str, int] = {}
    symbols = sorted(element_arrays)
    for first_index, first in enumerate(symbols):
        first_coords = element_arrays[first]
        for second in symbols[first_index:]:
            second_coords = element_arrays[second]
            histogram = torch.zeros(bins, dtype=torch.float64, device=device)
            accumulated_pairs = 0
            first_blocks = range(0, len(first_coords), block_size)
            second_blocks = range(0, len(second_coords), block_size)
            for first_start in first_blocks:
                first_stop = min(first_start + block_size, len(first_coords))
                first_block = first_coords[first_start:first_stop]
                for second_start in second_blocks:
                    if first == second and second_start < first_start:
                        continue
                    second_stop = min(second_start + block_size, len(second_coords))
                    second_block = second_coords[second_start:second_stop]
                    if first == second and second_start == first_start:
                        if len(first_block) < 2:
                            continue
                        distances = torch.pdist(first_block)
                    else:
                        distances = torch.cdist(first_block, second_block).reshape(-1)
                    accumulated_pairs += int(distances.numel())
                    histogram += torch.histc(
                        distances,
                        bins=bins,
                        min=0.0,
                        max=upper,
                    ).to(torch.float64)
                    del distances
            key = (first, second)
            histograms[key] = histogram.detach().cpu().numpy()
            pair_counts[f"{first}-{second}"] = accumulated_pairs
            del histogram
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "element_pair": f"{first}-{second}",
                        "pairs_accumulated": accumulated_pairs,
                        "bins": bins,
                    }
                ),
                flush=True,
            )
    metadata = {
        "block_size": block_size,
        "distance_bins": bins,
        "maximum_distance_angstrom": upper,
        "pair_counts": pair_counts,
    }
    return centers, histograms, counts, metadata


def chunked_debye_pattern(
    cluster: Cluster,
    two_theta: np.ndarray,
    *,
    wavelength: float,
    device: torch.device,
    bin_width: float,
    smooth_sigma_degrees: float,
    block_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    started = time.perf_counter()
    centers, histograms, counts, histogram_metadata = chunked_pair_distance_histograms(
        cluster,
        device=device,
        bin_width=bin_width,
        block_size=block_size,
    )
    theta = np.deg2rad(two_theta / 2.0)
    q_np = 4.0 * np.pi * np.sin(theta) / wavelength
    s2_np = (np.sin(theta) / wavelength) ** 2
    q = torch.as_tensor(q_np, dtype=torch.float32, device=device)
    s2 = torch.as_tensor(s2_np, dtype=torch.float32, device=device)
    radii = torch.as_tensor(centers, dtype=torch.float32, device=device)
    factors = {symbol: atomic_form_factor(symbol, s2) for symbol in counts}
    intensity = torch.zeros_like(q)
    for symbol, count in counts.items():
        intensity += count * factors[symbol] ** 2

    q_batch = 64
    for (first, second), histogram_np in histograms.items():
        histogram = torch.as_tensor(histogram_np, dtype=torch.float32, device=device)
        weighted_sinc = torch.empty_like(q)
        for start in range(0, len(q), q_batch):
            stop = min(start + q_batch, len(q))
            qr = q[start:stop, None] * radii[None, :]
            weighted_sinc[start:stop] = torch.sinc(qr / math.pi) @ histogram
        intensity += 2.0 * factors[first] * factors[second] * weighted_sinc

    raw = np.maximum(intensity.detach().cpu().numpy().astype(np.float64), 0.0)
    raw /= raw.max() if raw.max() > 0 else 1.0
    step = float(np.median(np.diff(two_theta)))
    sigma_points = smooth_sigma_degrees / step if smooth_sigma_degrees > 0 else 0.0
    smooth = gaussian_filter1d(raw, sigma=sigma_points) if sigma_points > 0 else raw.copy()
    smooth /= smooth.max() if smooth.max() > 0 else 1.0
    frame = pd.DataFrame(
        {
            "two_theta_deg": two_theta,
            "q_inverse_angstrom": q_np,
            "intensity_raw_normalized": raw,
            "intensity_smoothed_normalized": smooth,
        }
    )
    histogram_metadata["wall_seconds"] = time.perf_counter() - started
    histogram_metadata["element_counts"] = counts
    return frame, histogram_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mol2", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--reference-cif", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--two-theta-min", type=float, default=3.0)
    parser.add_argument("--two-theta-max", type=float, default=30.0)
    parser.add_argument("--two-theta-step", type=float, default=0.05)
    parser.add_argument("--wavelength", type=float, default=1.5406)
    parser.add_argument("--distance-bin", type=float, default=0.01)
    parser.add_argument("--smooth-sigma-degrees", type=float, default=0.08)
    parser.add_argument("--block-size", type=int, default=2048)
    args = parser.parse_args()
    if len(args.mol2) != len(args.labels):
        parser.error("--mol2 and --labels must have the same length")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    two_theta = np.arange(
        args.two_theta_min,
        args.two_theta_max + 0.5 * args.two_theta_step,
        args.two_theta_step,
    )
    records = []
    for path, label in zip(args.mol2, args.labels):
        cluster = parse_mol2(path)
        pattern, calculation = chunked_debye_pattern(
            cluster,
            two_theta,
            wavelength=args.wavelength,
            device=device,
            bin_width=args.distance_bin,
            smooth_sigma_degrees=args.smooth_sigma_degrees,
            block_size=args.block_size,
        )
        pattern.insert(0, "label", label)
        pattern.to_csv(args.output_dir / f"{label}_finite_debye_pxrd.csv", index=False)
        records.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "atoms": len(cluster.elements),
                "calculation": calculation,
            }
        )

    periodic_reference_peaks(
        args.reference_cif,
        (args.two_theta_min, args.two_theta_max),
        args.wavelength,
    ).to_csv(args.output_dir / "ideal_uio66_periodic_peak_positions.csv", index=False)
    metadata = {
        "schema_version": "large-cluster-chunked-debye-pxrd-v1",
        "device": str(device),
        "block_size": args.block_size,
        "distance_histogram_bin_angstrom": args.distance_bin,
        "two_theta_step_deg": args.two_theta_step,
        "patterns": records,
    }
    (args.output_dir / "pxrd_audit_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
