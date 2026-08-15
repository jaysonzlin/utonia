"""Encode one RGB PLY with Utonia and write a PCA-feature-colored PLY."""

import argparse
from pathlib import Path
from typing import Sequence


WAN_TRAINING_SEED = 42


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one cluster-safe Utonia PCA PLY visualization request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True, type=Path, help="Input RGB PLY file.")
    parser.add_argument(
        "--output", required=True, type=Path, help="Output PCA-colored PLY file."
    )
    parser.add_argument(
        "--checkpoint", help="Local Utonia checkpoint; defaults to Pointcept/Utonia."
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Utonia coordinate scale."
    )
    parser.add_argument(
        "--brightness", type=float, default=1.2, help="PCA display brightness."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=WAN_TRAINING_SEED,
        help="Random seed for Utonia grid sampling (default: Wan training seed 42).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Write Utonia PCA colors for every original point in an input PLY."""
    args = parse_args(argv)
    if args.scale <= 0.0:
        raise ValueError("--scale must be positive")
    if args.brightness <= 0.0:
        raise ValueError("--brightness must be positive")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run Utonia PCA PLY visualization")

    import numpy as np
    import open3d as o3d
    import utonia
    from utonia.joint_trajectory_pca import (
        build_object_input,
        load_utonia_model,
        move_tensors_to_cuda,
        pca_colors,
        upcast_features,
    )

    source = o3d.io.read_point_cloud(str(args.ply))
    coordinates = np.asarray(source.points, dtype=np.float32)
    colors = np.asarray(source.colors, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) == 0:
        raise ValueError(f"{args.ply} must contain at least one 3D point")
    if colors.shape != coordinates.shape:
        colors = np.zeros_like(coordinates)
    if not np.isfinite(coordinates).all() or not np.isfinite(colors).all():
        raise ValueError(f"{args.ply} contains non-finite coordinates or colors")

    # Open3D returns PLY colors in [0, 1]; Utonia's NormalizeColor divides by
    # 255, so restore the scale expected by its transform pipeline.
    utonia_colors = np.clip(colors, 0.0, 1.0) * 255.0
    utonia.utils.set_seed(args.seed)
    transform = utonia.transform.default(scale=args.scale, normalize_coord=True)
    point = transform(build_object_input(coordinates, utonia_colors))
    point = move_tensors_to_cuda(point)
    model = load_utonia_model(args.checkpoint)
    with torch.inference_mode():
        point = upcast_features(model(point))
        colors = pca_colors(point.feat, brightness=args.brightness)[point.inverse]

    output = o3d.geometry.PointCloud()
    output.points = o3d.utility.Vector3dVector(coordinates)
    output.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(args.output), output):
        raise RuntimeError(f"Failed to write PCA PLY: {args.output}")
    print(args.output)


if __name__ == "__main__":
    main()
