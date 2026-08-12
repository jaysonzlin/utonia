"""Export an object's RGB point-cloud trajectory as PLY files and an MP4 preview."""

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one object trajectory export request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=24)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Write every trajectory frame as RGB PLY and render a fixed-camera MP4."""
    args = parse_args(argv)
    if args.fps < 1:
        raise ValueError("fps must be a positive integer")

    from utonia.rgb_trajectory import (
        export_rgb_ply_sequence,
        export_rgb_trajectory_mp4,
        load_rgb_trajectory,
    )

    trajectory, rgb = load_rgb_trajectory(
        args.dataset_root, args.sample_id, args.object_id
    )
    ply_paths = export_rgb_ply_sequence(
        args.output_dir, args.sample_id, args.object_id, trajectory, rgb
    )
    mp4_path = export_rgb_trajectory_mp4(
        args.output_dir, args.sample_id, args.object_id, trajectory, rgb, args.fps
    )
    for path in [*ply_paths, mp4_path]:
        print(path)


if __name__ == "__main__":
    main()
