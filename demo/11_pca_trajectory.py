"""Export Utonia PCA-colored trajectory PLYs and a fixed-camera MP4 preview."""

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one CUDA-backed PCA trajectory visualization request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--checkpoint", help="local Utonia checkpoint; defaults to Pointcept/Utonia"
    )
    parser.add_argument("--seed", type=int, default=37)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Independently PCA-color every frame of one RGB point-cloud trajectory."""
    args = parse_args(argv)
    if args.fps < 1:
        raise ValueError("fps must be a positive integer")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run PCA trajectory visualization")

    import utonia
    from utonia.joint_trajectory_pca import load_utonia_model, move_tensors_to_cuda
    from utonia.pca_trajectory import (
        encode_pca_trajectory,
        export_pca_ply_sequence,
        export_pca_trajectory_mp4,
    )
    from utonia.rgb_trajectory import load_rgb_trajectory

    utonia.utils.set_seed(args.seed)
    model = load_utonia_model(args.checkpoint)
    transform = utonia.transform.default(scale=1.0, normalize_coord=True)
    trajectory, rgb = load_rgb_trajectory(
        args.dataset_root, args.sample_id, args.object_id
    )
    pca_colors_by_frame = encode_pca_trajectory(
        trajectory, rgb, transform, model, move_tensors_to_cuda
    )
    ply_paths = export_pca_ply_sequence(
        args.output_dir,
        args.sample_id,
        args.object_id,
        trajectory,
        pca_colors_by_frame,
    )
    mp4_path = export_pca_trajectory_mp4(
        args.output_dir,
        args.sample_id,
        args.object_id,
        trajectory,
        pca_colors_by_frame,
        args.fps,
    )
    for path in [*ply_paths, mp4_path]:
        print(path)


if __name__ == "__main__":
    main()
