"""Export Utonia PCA-colored initial clouds from a joint trajectory dataset."""

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the dataset and Utonia checkpoint selection arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument(
        "--start-frame",
        required=True,
        type=int,
        help="one-based trajectory conditioning frame",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--checkpoint",
        help="local Utonia checkpoint; defaults to Pointcept/Utonia",
    )
    parser.add_argument("--seed", type=int, default=37)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Encode and export every object in the selected training sample."""
    args = parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run joint trajectory PCA visualization")

    import utonia
    from utonia.joint_trajectory_pca import (
        build_object_input,
        export_object_pca,
        load_initial_point_clouds,
        load_utonia_model,
        move_tensors_to_cuda,
        pca_colors,
        upcast_features,
    )

    utonia.utils.set_seed(args.seed)
    model = load_utonia_model(args.checkpoint)
    transform = utonia.transform.default(scale=1.0, normalize_coord=True)
    for object_id, coords, rgb in load_initial_point_clouds(
        args.dataset_root, args.sample_id, args.start_frame
    ):
        point = transform(build_object_input(coords, rgb))
        point = move_tensors_to_cuda(point)
        with torch.inference_mode():
            point = upcast_features(model(point))
        colors = pca_colors(point.feat)[point.inverse].cpu().numpy()
        path = export_object_pca(
            args.output_dir,
            args.sample_id,
            args.start_frame - 1,
            object_id,
            coords,
            colors,
        )
        print(path)


if __name__ == "__main__":
    main()
