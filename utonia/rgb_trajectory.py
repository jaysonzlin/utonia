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


def trajectory_bounds(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return padded world-space bounds covering every frame of a trajectory."""
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
        raise ValueError("trajectory must have shape [T, N, 3]")
    if trajectory.shape[0] == 0 or trajectory.shape[1] == 0:
        raise ValueError("trajectory must contain at least one frame and one point")
    if not np.isfinite(trajectory).all():
        raise ValueError("trajectory coordinates must be finite")
    lower = np.min(trajectory, axis=(0, 1))
    upper = np.max(trajectory, axis=(0, 1))
    padding = np.maximum((upper - lower) * 0.05, 1e-6)
    return lower - padding, upper + padding


def export_rgb_trajectory_mp4(
    output_dir: Path | str,
    sample_id: str,
    object_id: str,
    trajectory: np.ndarray,
    rgb: np.ndarray,
    fps: int,
) -> Path:
    """Render a fixed-camera RGB trajectory preview as an MP4 file."""
    _require_basename(sample_id, "sample_id")
    _require_basename(object_id, "object_id")
    if fps < 1:
        raise ValueError("fps must be a positive integer")
    trajectory, rgb = _validate_trajectory_and_rgb(trajectory, rgb)
    lower, upper = trajectory_bounds(trajectory)

    try:
        import cv2
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "MP4 trajectory rendering requires matplotlib and opencv-python"
        ) from error

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sample_id}_object_{object_id}_trajectory.mp4"
    figure = plt.figure(figsize=(6.4, 6.4), dpi=100, facecolor="white")
    axes = figure.add_subplot(111, projection="3d")
    writer = None
    try:
        for frame_index, coords in enumerate(trajectory):
            axes.clear()
            axes.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=rgb,
                s=5,
                depthshade=False,
                linewidths=0,
            )
            axes.set_xlim(lower[0], upper[0])
            axes.set_ylim(lower[1], upper[1])
            axes.set_zlim(lower[2], upper[2])
            axes.set_box_aspect(tuple(upper - lower))
            axes.set_xlabel("x")
            axes.set_ylabel("y")
            axes.set_zlabel("z")
            axes.set_title(
                f"{sample_id} · object {object_id} · frame {frame_index}", pad=18
            )
            axes.view_init(elev=20, azim=-55)
            figure.canvas.draw()
            frame_rgba = np.asarray(figure.canvas.buffer_rgba())
            frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
            if writer is None:
                height, width = frame_bgr.shape[:2]
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise ValueError(f"unable to open MP4 writer: {path}")
            writer.write(frame_bgr)
    finally:
        if writer is not None:
            writer.release()
        plt.close(figure)
    return path
