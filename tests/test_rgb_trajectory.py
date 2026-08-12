import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "utonia" / "rgb_trajectory.py"
SPEC = importlib.util.spec_from_file_location("rgb_trajectory", MODULE_PATH)
rgb_trajectory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rgb_trajectory)


def load_demo_10():
    demo_path = Path(__file__).parents[1] / "demo" / "10_rgb_trajectory.py"
    demo_spec = importlib.util.spec_from_file_location("rgb_trajectory_demo", demo_path)
    demo = importlib.util.module_from_spec(demo_spec)
    demo_spec.loader.exec_module(demo)
    return demo


def frame_points(frame: int) -> np.ndarray:
    return np.array(
        [[frame, frame + 1, frame + 2], [frame + 3, frame + 4, frame + 5]],
        dtype=np.float32,
    )


def write_trajectory(root: Path, sample_id: str, object_id: str) -> None:
    path = root / sample_id / "objects" / object_id / "pc.hdf5"
    path.parent.mkdir(parents=True)
    cloud = np.stack([frame_points(0), frame_points(1)])[:, None]
    rgb = np.array([[0.0, 0.5, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    with h5py.File(path, "w") as source:
        source.create_dataset("point_cloud", data=cloud)
        source.create_dataset("rgb", data=rgb)


class RgbTrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_loads_one_object_trajectory_and_static_rgb(self) -> None:
        write_trajectory(self.root, "sample_0", "000")

        trajectory, rgb = rgb_trajectory.load_rgb_trajectory(
            self.root, "sample_0", "000"
        )

        self.assertEqual(trajectory.shape, (2, 2, 3))
        self.assertEqual(rgb.shape, (2, 3))
        np.testing.assert_array_equal(trajectory[1], frame_points(1))
        np.testing.assert_array_equal(
            rgb, np.array([[0.0, 0.5, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        )

    def test_export_writes_one_ascii_rgb_ply_per_frame(self) -> None:
        trajectory = np.stack([frame_points(0), frame_points(1)])
        rgb = np.array([[0.0, 0.5, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)

        paths = rgb_trajectory.export_rgb_ply_sequence(
            self.root, "sample_0", "000", trajectory, rgb
        )

        self.assertEqual(
            [path.name for path in paths],
            [
                "sample_0_object_000_frame_0000.ply",
                "sample_0_object_000_frame_0001.ply",
            ],
        )
        lines = paths[0].read_text().splitlines()
        self.assertEqual(lines[0], "ply")
        self.assertEqual(lines[10], "0.0 1.0 2.0 0 128 255")

    def test_trajectory_bounds_cover_every_frame_with_nonzero_extent(self) -> None:
        trajectory = np.array(
            [[[0.0, 0.0, 0.0]], [[2.0, 4.0, 6.0]]], dtype=np.float32
        )

        lower, upper = rgb_trajectory.trajectory_bounds(trajectory)

        self.assertTrue(np.all(lower < [0.0, 0.0, 0.0]))
        self.assertTrue(np.all(upper > [2.0, 4.0, 6.0]))

    def test_export_writes_a_nonempty_mp4(self) -> None:
        trajectory = np.stack([frame_points(0), frame_points(1)])
        rgb = np.array([[0.0, 0.5, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)

        path = rgb_trajectory.export_rgb_trajectory_mp4(
            self.root, "sample_0", "000", trajectory, rgb, fps=24
        )

        self.assertEqual(path.name, "sample_0_object_000_trajectory.mp4")
        self.assertGreater(path.stat().st_size, 0)

    def test_demo_defaults_to_24_fps(self) -> None:
        demo = load_demo_10()

        args = demo.parse_args(
            [
                "--dataset-root",
                "data/train",
                "--sample-id",
                "sample_0",
                "--object-id",
                "000",
                "--output-dir",
                "outputs/trajectory",
            ]
        )

        self.assertEqual(args.fps, 24)

    def test_demo_rejects_nonpositive_fps(self) -> None:
        demo = load_demo_10()

        with self.assertRaisesRegex(ValueError, "fps"):
            demo.main(
                [
                    "--dataset-root",
                    "data/train",
                    "--sample-id",
                    "sample_0",
                    "--object-id",
                    "000",
                    "--output-dir",
                    "outputs/trajectory",
                    "--fps",
                    "0",
                ]
            )


if __name__ == "__main__":
    unittest.main()
