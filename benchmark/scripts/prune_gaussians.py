#!/usr/bin/env python3
"""Offline pruning for 3D Gaussian PLY assets.

This keeps the highest-scoring Gaussians and writes standard 3DGS PLY files
that can be loaded by gaussian_renderer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gaussian_renderer.core.gaussiandata import GaussianData
from gaussian_renderer.core.util_gau import load_ply, save_ply


def gaussian_importance(g: GaussianData, mode: str) -> np.ndarray:
    opacity = np.asarray(g.opacity).reshape(-1)

    if mode == "opacity":
        return opacity

    if mode == "opacity_area":
        scale = np.asarray(g.scale)
        sorted_scale = np.sort(np.maximum(scale, 1e-8), axis=1)
        area = sorted_scale[:, 1] * sorted_scale[:, 2]
        return opacity * area

    if mode == "opacity_volume":
        scale = np.asarray(g.scale)
        volume = np.prod(np.maximum(scale, 1e-8), axis=1)
        return opacity * volume

    raise ValueError(f"unknown mode: {mode}")


def subset_gaussians(g: GaussianData, keep_idx: np.ndarray) -> GaussianData:
    return GaussianData(
        xyz=np.asarray(g.xyz)[keep_idx],
        rot=np.asarray(g.rot)[keep_idx],
        scale=np.asarray(g.scale)[keep_idx],
        opacity=np.asarray(g.opacity)[keep_idx],
        sh=np.asarray(g.sh)[keep_idx],
    )


def prune_file(
    src: Path,
    dst: Path,
    keep_ratio: float | None,
    keep_count: int | None,
    mode: str,
) -> tuple[int, int]:
    g = load_ply(src)
    before = len(g)

    if keep_count is None:
        assert keep_ratio is not None
        keep_count = int(round(before * keep_ratio))
    keep_count = min(max(1, keep_count), before)

    score = gaussian_importance(g, mode)
    keep_idx = np.argpartition(score, -keep_count)[-keep_count:]
    keep_idx = keep_idx[np.argsort(score[keep_idx])[::-1]]

    save_ply(subset_gaussians(g, keep_idx), dst)
    return before, keep_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input PLY file or directory")
    parser.add_argument("--output", type=Path, required=True, help="Output PLY file or directory")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--keep-ratio", type=float, help="Fraction of points to keep, e.g. 0.1")
    group.add_argument("--keep-count", type=int, help="Number of points to keep per file")

    parser.add_argument(
        "--mode",
        choices=["opacity", "opacity_area", "opacity_volume"],
        default="opacity_area",
        help="Importance score used for top-k pruning",
    )
    args = parser.parse_args()

    if args.keep_ratio is not None and not (0.0 < args.keep_ratio <= 1.0):
        raise ValueError("--keep-ratio must be in (0, 1]")

    if args.input.is_file():
        output = args.output
        if output.suffix.lower() != ".ply":
            output = output / args.input.name
        before, after = prune_file(args.input, output, args.keep_ratio, args.keep_count, args.mode)
        print(f"{args.input} -> {output}: {before} -> {after} ({after / before:.1%})")
        return

    if not args.input.is_dir():
        raise FileNotFoundError(args.input)

    total_before = 0
    total_after = 0
    for src in sorted(args.input.rglob("*.ply")):
        rel = src.relative_to(args.input)
        dst = args.output / rel
        before, after = prune_file(src, dst, args.keep_ratio, args.keep_count, args.mode)
        total_before += before
        total_after += after
        print(f"{rel}: {before} -> {after} ({after / before:.1%})")

    print(f"total: {total_before} -> {total_after} ({total_after / total_before:.1%})")


if __name__ == "__main__":
    main()
