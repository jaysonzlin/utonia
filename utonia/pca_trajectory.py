"""Frame-wise Utonia PCA export helpers for RGB point-cloud trajectories."""

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from utonia.joint_trajectory_pca import (
    build_object_input,
    pca_colors,
    upcast_features,
)
from utonia.rgb_trajectory import (
    _require_basename,
    _validate_trajectory_and_rgb,
    export_colored_trajectory_mp4,
)


def _validate_pca_colors(
    trajectory: np.ndarray, pca_colors_by_frame: np.ndarray
) -> np.ndarray:
    pca_colors_by_frame = np.asarray(pca_colors_by_frame, dtype=np.float32)
    if pca_colors_by_frame.shape != trajectory.shape:
        raise ValueError(
            "pca_colors_by_frame must have shape "
            f"{trajectory.shape}; got {pca_colors_by_frame.shape}"
        )
    if not np.isfinite(pca_colors_by_frame).all() or np.any(
        (pca_colors_by_frame < 0.0) | (pca_colors_by_frame > 1.0)
    ):
        raise ValueError("pca_colors_by_frame must be finite and in [0, 1]")
    return pca_colors_by_frame


def encode_pca_trajectory(
    trajectory: np.ndarray,
    rgb: np.ndarray,
    transform: Callable,
    model: Callable,
    move_to_cuda: Callable,
) -> np.ndarray:
    """Encode each trajectory frame and independently map it to PCA RGB."""
    trajectory, rgb = _validate_trajectory_and_rgb(trajectory, rgb)
    colors_by_frame = []
    for coords in trajectory:
        point = move_to_cuda(transform(build_object_input(coords, rgb)))
        with torch.inference_mode():
            point = upcast_features(model(point))
        colors = pca_colors(point.feat)[point.inverse].cpu().numpy()
        colors_by_frame.append(colors)
    return _validate_pca_colors(trajectory, np.stack(colors_by_frame))


def _write_pca_ply(path: Path, coords: np.ndarray, colors: np.ndarray) -> None:
    color_bytes = np.rint(np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
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


def export_pca_ply_sequence(
    output_dir: Path | str,
    sample_id: str,
    object_id: str,
    trajectory: np.ndarray,
    pca_colors_by_frame: np.ndarray,
) -> list[Path]:
    """Export one independently PCA-colored ASCII PLY per trajectory frame."""
    _require_basename(sample_id, "sample_id")
    _require_basename(object_id, "object_id")
    trajectory = np.asarray(trajectory, dtype=np.float32)
    pca_colors_by_frame = _validate_pca_colors(trajectory, pca_colors_by_frame)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for frame_index, (coords, colors) in enumerate(
        zip(trajectory, pca_colors_by_frame)
    ):
        path = output_dir / (
            f"{sample_id}_object_{object_id}_frame_{frame_index:04d}_pca.ply"
        )
        _write_pca_ply(path, coords, colors)
        paths.append(path)
    return paths


def export_pca_trajectory_mp4(
    output_dir: Path | str,
    sample_id: str,
    object_id: str,
    trajectory: np.ndarray,
    pca_colors_by_frame: np.ndarray,
    fps: int,
) -> Path:
    """Render an independently PCA-colored trajectory as a fixed-camera MP4."""
    _require_basename(sample_id, "sample_id")
    _require_basename(object_id, "object_id")
    trajectory = np.asarray(trajectory, dtype=np.float32)
    pca_colors_by_frame = _validate_pca_colors(trajectory, pca_colors_by_frame)
    return export_colored_trajectory_mp4(
        output_dir,
        f"{sample_id}_object_{object_id}_pca_trajectory.mp4",
        f"{sample_id} · object {object_id} · PCA",
        trajectory,
        pca_colors_by_frame,
        fps,
    )
