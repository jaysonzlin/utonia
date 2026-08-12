"""Export static-RGB point-cloud trajectories without requiring Utonia inference."""

from pathlib import Path

import h5py
import numpy as np


def _require_basename(value: str, name: str) -> None:
    path = Path(value)
    if path.is_absolute() or path.name != value:
        raise ValueError(f"{name} must be a directory name without path separators")


def _validate_trajectory_and_rgb(
    trajectory: np.ndarray, rgb: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    trajectory = np.asarray(trajectory, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
        raise ValueError("trajectory must have shape [T, N, 3]")
    if trajectory.shape[0] == 0 or trajectory.shape[1] == 0:
        raise ValueError("trajectory must contain at least one frame and one point")
    if rgb.shape != trajectory.shape[1:]:
        raise ValueError(
            f"rgb must have shape {trajectory.shape[1:]}; got {rgb.shape}"
        )
    if not np.isfinite(trajectory).all():
        raise ValueError("trajectory coordinates must be finite")
    if not np.isfinite(rgb).all() or np.any((rgb < 0.0) | (rgb > 1.0)):
        raise ValueError("rgb values must be finite and in [0, 1]")
    return trajectory, rgb


def load_rgb_trajectory(
    dataset_root: Path | str, sample_id: str, object_id: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load one object's `[T, N, 3]` coordinates and static `[N, 3]` RGB."""
    _require_basename(sample_id, "sample_id")
    _require_basename(object_id, "object_id")
    path = Path(dataset_root) / sample_id / "objects" / object_id / "pc.hdf5"
    if not path.is_file():
        raise FileNotFoundError(f"missing point cloud file: {path}")

    try:
        with h5py.File(path, "r") as source:
            for dataset_name in ("point_cloud", "rgb"):
                if dataset_name not in source:
                    raise ValueError(f"{path}: missing {dataset_name} dataset")
                if not isinstance(source[dataset_name], h5py.Dataset):
                    raise ValueError(f"{path}: {dataset_name} must be an HDF5 dataset")
            point_cloud = source["point_cloud"]
            if (
                point_cloud.ndim != 4
                or point_cloud.shape[1] != 1
                or point_cloud.shape[-1] != 3
            ):
                raise ValueError(
                    f"{path}: point_cloud must have shape [T, 1, N, 3]; "
                    f"got {point_cloud.shape}"
                )
            try:
                trajectory = np.asarray(point_cloud[:, 0], dtype=np.float32)
                rgb = np.asarray(source["rgb"], dtype=np.float32)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}: point_cloud and rgb must be numeric") from error
    except OSError as error:
        raise ValueError(f"invalid HDF5 point cloud: {path}") from error
    return _validate_trajectory_and_rgb(trajectory, rgb)


def _write_rgb_ply(path: Path, coords: np.ndarray, rgb: np.ndarray) -> None:
    color_bytes = np.rint(np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    header = "\n".join(
        (
            "ply",
            "format ascii 1.0",
            f"element vertex {len(coords)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        )
    )
    vertices = "\n".join(
        f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}"
        for point, color in zip(coords, color_bytes)
    )
    path.write_text(f"{header}\n{vertices}\n")


def export_rgb_ply_sequence(
    output_dir: Path | str,
    sample_id: str,
    object_id: str,
    trajectory: np.ndarray,
    rgb: np.ndarray,
) -> list[Path]:
    """Export one ASCII RGB PLY per frame of a validated trajectory."""
    _require_basename(sample_id, "sample_id")
    _require_basename(object_id, "object_id")
    trajectory, rgb = _validate_trajectory_and_rgb(trajectory, rgb)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for frame_index, coords in enumerate(trajectory):
        path = output_dir / f"{sample_id}_object_{object_id}_frame_{frame_index:04d}.ply"
        _write_rgb_ply(path, coords, rgb)
        paths.append(path)
    return paths
