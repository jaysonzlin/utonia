import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rgb_trajectory = load_module("rgb_trajectory", ROOT / "utonia" / "rgb_trajectory.py")
joint_trajectory_pca = load_module(
    "joint_trajectory_pca", ROOT / "utonia" / "joint_trajectory_pca.py"
)
utonia_package = types.ModuleType("utonia")
utonia_package.__path__ = []
sys.modules["utonia"] = utonia_package
sys.modules["utonia.rgb_trajectory"] = rgb_trajectory
sys.modules["utonia.joint_trajectory_pca"] = joint_trajectory_pca
pca_trajectory = load_module("pca_trajectory", ROOT / "utonia" / "pca_trajectory.py")


def frame_points(frame: int) -> np.ndarray:
    return np.array(
        [[frame, frame + 1, frame + 2], [frame + 3, frame + 4, frame + 5]],
        dtype=np.float32,
    )


class FakePoint(dict):
    def __init__(self, feat: torch.Tensor) -> None:
        super().__init__()
        self.feat = feat
        self.inverse = torch.tensor([0, 1])


class PcaTrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.trajectory = np.stack([frame_points(0), frame_points(1)])
        self.rgb = np.full((2, 3), 0.25, dtype=np.float32)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_encodes_each_frame_with_static_rgb_and_independent_pca(self) -> None:
        captured_colors = []

        def transform(point):
            captured_colors.append(point["color"].copy())
            return FakePoint(torch.arange(12, dtype=torch.float32).reshape(2, 6))

        with patch.object(
            pca_trajectory,
            "pca_colors",
            side_effect=[torch.zeros((2, 3)), torch.ones((2, 3))],
        ) as mapping, patch.object(
            pca_trajectory, "upcast_features", side_effect=lambda point: point
        ):
            colors = pca_trajectory.encode_pca_trajectory(
                self.trajectory, self.rgb, transform, lambda point: point, lambda point: point
            )

        self.assertEqual(mapping.call_count, 2)
        np.testing.assert_array_equal(captured_colors[0], self.rgb)
        np.testing.assert_array_equal(captured_colors[1], self.rgb)
        np.testing.assert_array_equal(colors[0], np.zeros((2, 3)))
        np.testing.assert_array_equal(colors[1], np.ones((2, 3)))

    def test_pca_ply_sequence_and_mp4_have_pca_names(self) -> None:
        colors = np.stack([np.zeros((2, 3)), np.ones((2, 3))]).astype(np.float32)

        plys = pca_trajectory.export_pca_ply_sequence(
            self.root, "sample_0", "000", self.trajectory, colors
        )
        mp4 = pca_trajectory.export_pca_trajectory_mp4(
            self.root, "sample_0", "000", self.trajectory, colors, 24
        )

        self.assertEqual(plys[0].name, "sample_0_object_000_frame_0000_pca.ply")
        self.assertEqual(mp4.name, "sample_0_object_000_pca_trajectory.mp4")
        self.assertGreater(mp4.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
