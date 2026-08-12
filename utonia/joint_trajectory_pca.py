"""Helpers for visualizing joint-training trajectory point clouds with Utonia."""

from pathlib import Path

import h5py
import numpy as np
import torch


def load_initial_point_clouds(
    dataset_root: Path | str, sample_id: str, start_frame: int
) -> list[tuple[str, np.ndarray]]:
    """Load each object's training-condition cloud for a one-based frame."""
    if start_frame < 1:
        raise ValueError("start_frame must be one-based and at least 1")

    sample_path = Path(sample_id)
    if sample_path.is_absolute() or sample_path.name != sample_id:
        raise ValueError("sample_id must be a directory name without path separators")

    sample_dir = Path(dataset_root) / sample_id
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"missing sample directory: {sample_dir}")
    object_root = sample_dir / "objects"
    if not object_root.is_dir():
        raise FileNotFoundError(f"missing objects directory: {object_root}")

    object_dirs = sorted(path for path in object_root.iterdir() if path.is_dir())
    if not object_dirs:
        raise ValueError(f"no object directories found in {object_root}")

    selected_clouds = []
    for object_dir in object_dirs:
        path = object_dir / "pc.hdf5"
        if not path.is_file():
            raise FileNotFoundError(f"missing point cloud file: {path}")
        try:
            with h5py.File(path, "r") as source:
                if "point_cloud" not in source:
                    raise ValueError(f"{path}: missing point_cloud dataset")
                cloud = source["point_cloud"]
                if not isinstance(cloud, h5py.Dataset):
                    raise ValueError(f"{path}: point_cloud must be an HDF5 dataset")
                if cloud.ndim != 4 or cloud.shape[1] != 1 or cloud.shape[-1] != 3:
                    raise ValueError(
                        f"{path}: point_cloud must have shape [T, 1, N, 3]"
                    )
                if start_frame > cloud.shape[0]:
                    raise ValueError(
                        f"{path}: start frame {start_frame} exceeds {cloud.shape[0]} frames"
                    )
                try:
                    coords = np.asarray(cloud[start_frame - 1, 0], dtype=np.float32)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{path}: point_cloud must contain numeric coordinates"
                    ) from error
        except OSError as error:
            raise ValueError(f"invalid HDF5 point cloud: {path}") from error
        selected_clouds.append((object_dir.name, coords))
    return selected_clouds


def build_object_input(coords: np.ndarray) -> dict[str, np.ndarray]:
    """Build Utonia's coordinate-only input with absent modalities zero-filled."""
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape [N, 3]")
    return {
        "coord": coords.copy(),
        "color": np.zeros_like(coords),
        "normal": np.zeros_like(coords),
    }


def pca_colors(features: torch.Tensor, brightness: float = 1.2) -> torch.Tensor:
    """Apply Utonia's object-demo PCA mapping to one object's features."""
    if features.ndim != 2 or features.shape[1] < 6:
        raise ValueError("features must have shape [N, D] with D >= 6")
    if features.shape[0] < 6:
        raise ValueError("at least 6 points are required for PCA visualization")

    _, _, basis = torch.pca_lowrank(
        features, center=True, niter=5, q=min(9, *features.shape)
    )
    projection = features @ basis
    projection = projection[:, :3] * 0.6 + projection[:, 3:6] * 0.4
    minimum = projection.min(dim=0, keepdim=True).values
    maximum = projection.max(dim=0, keepdim=True).values
    return (
        (projection - minimum)
        / (maximum - minimum).clamp_min(1e-6)
        * brightness
    ).clamp(0.0, 1.0)


def require_cuda() -> str:
    """Return the CUDA device name or reject unsupported local execution."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run joint trajectory PCA visualization")
    return "cuda"


def export_object_pca(
    output_dir: Path | str,
    sample_id: str,
    raw_frame: int,
    object_id: str,
    coords: np.ndarray,
    colors: np.ndarray,
) -> Path:
    """Write original coordinates and RGB colors in an ASCII PLY file."""
    coords = np.asarray(coords, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape [N, 3]")
    if colors.shape != coords.shape:
        raise ValueError("colors must have shape [N, 3] matching coords")

    path = Path(output_dir) / (
        f"{sample_id}_frame_{raw_frame:04d}_{object_id}_pca.ply"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.rint(np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
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
        for point, color in zip(coords, rgb)
    )
    path.write_text(f"{header}\n{vertices}\n")
    return path


def move_tensors_to_cuda(point: dict) -> dict:
    """Move the tensors in a transformed Utonia point dictionary to CUDA."""
    return {
        key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in point.items()
    }


def upcast_features(point):
    """Restore encoded features through Utonia's pooling hierarchy."""
    for _ in range(2):
        if "pooling_parent" not in point or "pooling_inverse" not in point:
            raise ValueError("Utonia output is missing required pooling hierarchy")
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    while "pooling_parent" in point:
        if "pooling_inverse" not in point:
            raise ValueError("Utonia output is missing required pooling inverse")
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent
    return point


def load_utonia_model(checkpoint: str | None):
    """Load Utonia on CUDA with its existing non-FlashAttention fallback."""
    device = require_cuda()
    import utonia

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        load_kwargs = {
            "custom_config": {
                "enc_patch_size": [1024 for _ in range(5)],
                "enable_flash": False,
            }
        }
    else:
        load_kwargs = {}

    if checkpoint is None:
        model = utonia.load("utonia", repo_id="Pointcept/Utonia", **load_kwargs)
    else:
        model = utonia.load(checkpoint, **load_kwargs)
    return model.to(device).eval()
