import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch


MODULE_PATH = Path(__file__).parents[1] / "utonia" / "joint_trajectory_pca.py"
SPEC = importlib.util.spec_from_file_location("joint_trajectory_pca", MODULE_PATH)
joint_trajectory_pca = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(joint_trajectory_pca)


def frame_points(offset: int, frame: int) -> np.ndarray:
    return np.array(
        [
            [offset + frame, offset + frame + 1, offset + frame + 2],
            [offset + frame + 3, offset + frame + 4, offset + frame + 5],
        ],
        dtype=np.float32,
    )


def write_cloud(
    root: Path,
    sample_id: str,
    object_id: str,
    frames: int,
    offset: int,
    rgb: np.ndarray | None = None,
) -> None:
    path = root / sample_id / "objects" / object_id / "pc.hdf5"
    path.parent.mkdir(parents=True)
    cloud = np.stack([frame_points(offset, frame) for frame in range(frames)])[:, None]
    with h5py.File(path, "w") as source:
        source.create_dataset("point_cloud", data=cloud)
        source.create_dataset(
            "rgb", data=np.zeros((2, 3), dtype=np.float32) if rgb is None else rgb
        )


def write_invalid_cloud(root: Path, sample_id: str, object_id: str, shape: tuple[int, ...]) -> None:
    path = root / sample_id / "objects" / object_id / "pc.hdf5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as source:
        source.create_dataset("point_cloud", data=np.zeros(shape, dtype=np.float32))


class LoadInitialPointCloudsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_loads_sorted_objects_at_the_requested_condition_frame(self) -> None:
        write_cloud(self.root, "sample-a", "object-b", frames=3, offset=100)
        write_cloud(self.root, "sample-a", "object-a", frames=3, offset=0)

        loaded = joint_trajectory_pca.load_initial_point_clouds(
            self.root, "sample-a", start_frame=2
        )

        self.assertEqual([name for name, *_ in loaded], ["object-a", "object-b"])
        np.testing.assert_array_equal(loaded[0][1], frame_points(0, 1))
        np.testing.assert_array_equal(loaded[1][1], frame_points(100, 1))

    def test_loads_static_rgb_aligned_with_each_object_cloud(self) -> None:
        rgb = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        write_cloud(self.root, "sample-a", "object-a", frames=3, offset=0, rgb=rgb)

        loaded = joint_trajectory_pca.load_initial_point_clouds(
            self.root, "sample-a", start_frame=2
        )

        object_id, coords, loaded_rgb = loaded[0]
        self.assertEqual(object_id, "object-a")
        np.testing.assert_array_equal(coords, frame_points(0, 1))
        np.testing.assert_array_equal(loaded_rgb, rgb)

    def test_rejects_rgb_with_a_different_point_count(self) -> None:
        write_cloud(
            self.root,
            "sample-a",
            "object-a",
            frames=2,
            offset=0,
            rgb=np.zeros((3, 3), dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "rgb must have shape"):
            joint_trajectory_pca.load_initial_point_clouds(
                self.root, "sample-a", start_frame=1
            )

    def test_rejects_out_of_range_start_frame(self) -> None:
        write_cloud(self.root, "sample-a", "object-a", frames=2, offset=0)

        with self.assertRaisesRegex(ValueError, r"start frame 3.*2"):
            joint_trajectory_pca.load_initial_point_clouds(
                self.root, "sample-a", start_frame=3
            )

    def test_rejects_non_training_point_cloud_shape(self) -> None:
        write_invalid_cloud(self.root, "sample-a", "object-a", shape=(2, 4, 3))

        with self.assertRaisesRegex(ValueError, r"\[T, 1, N, 3\]"):
            joint_trajectory_pca.load_initial_point_clouds(
                self.root, "sample-a", start_frame=1
            )

    def test_rejects_a_missing_sample_before_looking_for_objects(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "missing sample directory"):
            joint_trajectory_pca.load_initial_point_clouds(
                self.root, "sample-a", start_frame=1
            )

    def test_rejects_path_like_sample_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_id must be a directory name"):
            joint_trajectory_pca.load_initial_point_clouds(
                self.root, "../sample-a", start_frame=1
            )

    def test_rejects_a_corrupt_hdf5_file_with_its_path(self) -> None:
        path = self.root / "sample-a" / "objects" / "object-a" / "pc.hdf5"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not an HDF5 file")

        with self.assertRaisesRegex(ValueError, r"invalid HDF5 point cloud.*pc\.hdf5"):
            joint_trajectory_pca.load_initial_point_clouds(
                self.root, "sample-a", start_frame=1
            )

    def test_rejects_a_point_cloud_group(self) -> None:
        path = self.root / "sample-a" / "objects" / "object-a" / "pc.hdf5"
        path.parent.mkdir(parents=True)
        with h5py.File(path, "w") as source:
            source.create_group("point_cloud")

        with self.assertRaisesRegex(ValueError, "must be an HDF5 dataset"):
            joint_trajectory_pca.load_initial_point_clouds(
                self.root, "sample-a", start_frame=1
            )

    def test_rejects_non_numeric_point_coordinates(self) -> None:
        path = self.root / "sample-a" / "objects" / "object-a" / "pc.hdf5"
        path.parent.mkdir(parents=True)
        with h5py.File(path, "w") as source:
            source.create_dataset(
                "point_cloud", data=np.full((1, 1, 2, 3), b"bad", dtype="S3")
            )

        with self.assertRaisesRegex(ValueError, "must contain numeric coordinates"):
            joint_trajectory_pca.load_initial_point_clouds(
                self.root, "sample-a", start_frame=1
            )


class JointTrajectoryPcaHelperTests(unittest.TestCase):
    def test_build_object_input_zero_fills_missing_modalities(self) -> None:
        coords = np.arange(12, dtype=np.float32).reshape(4, 3)

        point = joint_trajectory_pca.build_object_input(coords)

        np.testing.assert_array_equal(point["coord"], coords)
        np.testing.assert_array_equal(point["color"], np.zeros_like(coords))
        np.testing.assert_array_equal(point["normal"], np.zeros_like(coords))

    def test_build_object_input_preserves_rgb(self) -> None:
        coords = np.arange(12, dtype=np.float32).reshape(4, 3)
        rgb = np.full((4, 3), 0.25, dtype=np.float32)

        point = joint_trajectory_pca.build_object_input(coords, rgb)

        np.testing.assert_array_equal(point["color"], rgb)

    def test_pca_colors_returns_bounded_rgb_for_one_object(self) -> None:
        features = torch.arange(144, dtype=torch.float32).reshape(12, 12)

        colors = joint_trajectory_pca.pca_colors(features)

        self.assertEqual(tuple(colors.shape), (12, 3))
        self.assertTrue(torch.all((0 <= colors) & (colors <= 1)))

    def test_pca_colors_supports_the_smallest_valid_feature_matrix(self) -> None:
        features = torch.arange(36, dtype=torch.float32).reshape(6, 6)

        colors = joint_trajectory_pca.pca_colors(features)

        self.assertEqual(tuple(colors.shape), (6, 3))
        self.assertTrue(torch.all((0 <= colors) & (colors <= 1)))


class JointTrajectoryPcaExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_require_cuda_fails_with_an_actionable_message(self) -> None:
        original_is_available = joint_trajectory_pca.torch.cuda.is_available
        joint_trajectory_pca.torch.cuda.is_available = lambda: False
        self.addCleanup(
            setattr, joint_trajectory_pca.torch.cuda, "is_available", original_is_available
        )

        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            joint_trajectory_pca.require_cuda()

    def test_export_writes_original_coordinates_and_pca_rgb(self) -> None:
        coords = np.arange(12, dtype=np.float32).reshape(4, 3)
        colors = np.full((4, 3), 0.5, dtype=np.float32)

        path = joint_trajectory_pca.export_object_pca(
            self.root, "sample-a", 12, "object-a", coords, colors
        )

        self.assertEqual(path.name, "sample-a_frame_0012_object-a_pca.ply")
        lines = path.read_text().splitlines()
        self.assertEqual(lines[0], "ply")
        self.assertEqual(lines[10], "0.0 1.0 2.0 128 128 128")

    def test_upcast_features_restores_features_through_pooling_parents(self) -> None:
        class FakePoint(dict):
            def __init__(self, feat: torch.Tensor, **values: object) -> None:
                super().__init__(values)
                self.feat = feat

        root = FakePoint(torch.ones(2, 1))
        middle = FakePoint(torch.full((2, 1), 2.0), pooling_parent=root, pooling_inverse=torch.tensor([0, 1]))
        leaf = FakePoint(torch.full((2, 1), 3.0), pooling_parent=middle, pooling_inverse=torch.tensor([1, 0]))

        restored = joint_trajectory_pca.upcast_features(leaf)

        self.assertIs(restored, root)
        torch.testing.assert_close(
            restored.feat,
            torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        )


class JointTrajectoryPcaCliTests(unittest.TestCase):
    def _load_demo(self):
        demo_path = Path(__file__).parents[1] / "demo" / "9_pca_joint_trajectory.py"
        spec = importlib.util.spec_from_file_location("joint_trajectory_demo", demo_path)
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)
        return demo

    def test_parse_args_requires_training_dataset_identifiers(self) -> None:
        demo = self._load_demo()

        args = demo.parse_args(
            [
                "--dataset-root",
                "data/train",
                "--sample-id",
                "000001",
                "--start-frame",
                "13",
                "--output-dir",
                "outputs/pca",
            ]
        )

        self.assertEqual(args.start_frame, 13)
        self.assertEqual(args.seed, 37)
        self.assertIsNone(args.checkpoint)

    @patch("torch.cuda.is_available", return_value=False)
    def test_main_rejects_a_non_cuda_host_before_importing_utonia(self, _available) -> None:
        demo = self._load_demo()

        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            demo.main(
                [
                    "--dataset-root",
                    "data/train",
                    "--sample-id",
                    "000001",
                    "--start-frame",
                    "13",
                    "--output-dir",
                    "outputs/pca",
                ]
            )
