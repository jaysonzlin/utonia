# RGB Trajectory PLY and Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Demo 10 to export every frame of one RGB point-cloud trajectory as PLY files and a fixed-camera MP4 preview.

**Architecture:** Put HDF5 validation, PLY serialization, fixed-bound calculation, and MP4 rendering in `utonia/rgb_trajectory.py`. Keep `demo/10_rgb_trajectory.py` as a thin argument-parsing and orchestration layer. The helper consumes the existing object-level `pc.hdf5` layout and has no model or CUDA dependency.

**Tech Stack:** Python 3.10, NumPy, h5py, Matplotlib Agg, OpenCV VideoWriter, unittest.

## Global Constraints

- Do not modify `environment.yml` or `utonia.def`.
- Do not load a Utonia checkpoint or require CUDA.
- Require `point_cloud` shape `[T, 1, N, 3]` and static `rgb` shape `[N, 3]` with finite float values in `[0, 1]`.
- Reject path-like `sample_id` and `object_id` values.
- Write ASCII PLY files named `<sample-id>_object_<object-id>_frame_<t:04d>.ply` with original coordinates and `uchar` RGB.
- Write `<sample-id>_object_<object-id>_trajectory.mp4` at 24 FPS by default, with a positive `--fps` override.
- Use one world-space bound and one fixed camera for every MP4 frame.

---

## File structure

- Create `utonia/rgb_trajectory.py`: input validation, PLY export, fixed-bound rendering, and MP4 export.
- Create `demo/10_rgb_trajectory.py`: command-line parser and calls into `utonia.rgb_trajectory`.
- Create `tests/test_rgb_trajectory.py`: real temporary HDF5/PLY/MP4 behavior tests.

### Task 1: Build the validated trajectory and PLY export helper

**Files:**

- Create: `utonia/rgb_trajectory.py`
- Create: `tests/test_rgb_trajectory.py`

**Interfaces:**

- Produces `load_rgb_trajectory(dataset_root: Path | str, sample_id: str, object_id: str) -> tuple[np.ndarray, np.ndarray]`, returning coordinates `[T, N, 3]` and static colors `[N, 3]` as float32.
- Produces `export_rgb_ply_sequence(output_dir: Path | str, sample_id: str, object_id: str, trajectory: np.ndarray, rgb: np.ndarray) -> list[Path]`.
- Consumes the object file at `<dataset-root>/<sample-id>/objects/<object-id>/pc.hdf5`.

- [ ] **Step 1: Write the failing tests**

```python
def test_loads_one_object_trajectory_and_static_rgb(self):
    write_trajectory(self.root, "sample_0", "000")

    trajectory, rgb = rgb_trajectory.load_rgb_trajectory(
        self.root, "sample_0", "000"
    )

    self.assertEqual(trajectory.shape, (2, 2, 3))
    self.assertEqual(rgb.shape, (2, 3))
    np.testing.assert_array_equal(trajectory[1], frame_points(1))


def test_export_writes_one_ascii_rgb_ply_per_frame(self):
    trajectory = np.stack([frame_points(0), frame_points(1)])
    rgb = np.array([[0.0, 0.5, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    paths = rgb_trajectory.export_rgb_ply_sequence(
        self.root, "sample_0", "000", trajectory, rgb
    )

    self.assertEqual([path.name for path in paths], [
        "sample_0_object_000_frame_0000.ply",
        "sample_0_object_000_frame_0001.ply",
    ])
    self.assertIn("0 128 255", paths[0].read_text())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n utonia-dev python -m unittest tests.test_rgb_trajectory -v`

Expected: FAIL because `utonia/rgb_trajectory.py` does not exist.

- [ ] **Step 3: Implement the minimal helper**

```python
def load_rgb_trajectory(dataset_root, sample_id, object_id):
    # Validate basename-only IDs, HDF5 datasets, shapes, finite coordinates,
    # numeric RGB, and [0, 1] RGB values; return cloud[:, 0] and rgb.


def export_rgb_ply_sequence(output_dir, sample_id, object_id, trajectory, rgb):
    # Convert rgb once with round(255 * clip(rgb, 0, 1)), then write one ASCII
    # PLY with x/y/z/red/green/blue properties for each trajectory frame.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n utonia-dev python -m unittest tests.test_rgb_trajectory -v`

Expected: PASS.

- [ ] **Step 5: Commit the helper and tests**

```bash
git add utonia/rgb_trajectory.py tests/test_rgb_trajectory.py
git commit -m "feat: export RGB trajectory PLY sequence"
```

### Task 2: Add fixed-camera MP4 rendering

**Files:**

- Modify: `utonia/rgb_trajectory.py`
- Modify: `tests/test_rgb_trajectory.py`

**Interfaces:**

- Consumes `trajectory: np.ndarray` with shape `[T, N, 3]` and `rgb: np.ndarray` with shape `[N, 3]` from Task 1.
- Produces `trajectory_bounds(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]`.
- Produces `export_rgb_trajectory_mp4(output_dir: Path | str, sample_id: str, object_id: str, trajectory: np.ndarray, rgb: np.ndarray, fps: int) -> Path`.

- [ ] **Step 1: Write the failing tests**

```python
def test_trajectory_bounds_cover_every_frame_with_nonzero_extent(self):
    trajectory = np.array([
        [[0.0, 0.0, 0.0]],
        [[2.0, 4.0, 6.0]],
    ], dtype=np.float32)

    lower, upper = rgb_trajectory.trajectory_bounds(trajectory)

    self.assertTrue(np.all(lower < [0.0, 0.0, 0.0]))
    self.assertTrue(np.all(upper > [2.0, 4.0, 6.0]))


def test_export_writes_a_nonempty_mp4(self):
    path = rgb_trajectory.export_rgb_trajectory_mp4(
        self.root, "sample_0", "000", self.trajectory, self.rgb, fps=24
    )

    self.assertEqual(path.name, "sample_0_object_000_trajectory.mp4")
    self.assertGreater(path.stat().st_size, 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n utonia-dev python -m unittest tests.test_rgb_trajectory -v`

Expected: FAIL because the bounds and MP4 functions do not exist.

- [ ] **Step 3: Implement fixed-bound rendering**

```python
def trajectory_bounds(trajectory):
    lower = np.min(trajectory, axis=(0, 1))
    upper = np.max(trajectory, axis=(0, 1))
    padding = np.maximum((upper - lower) * 0.05, 1e-6)
    return lower - padding, upper + padding


def export_rgb_trajectory_mp4(output_dir, sample_id, object_id, trajectory, rgb, fps):
    # Reject fps < 1. Force the Agg backend before importing pyplot.
    # Render each frame with identical limits, aspect, elevation, azimuth,
    # white background, and RGB scatter points; convert RGB to BGR and append
    # to cv2.VideoWriter using mp4v. Raise ValueError if the writer cannot open.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n utonia-dev python -m unittest tests.test_rgb_trajectory -v`

Expected: PASS.

- [ ] **Step 5: Commit the renderer and tests**

```bash
git add utonia/rgb_trajectory.py tests/test_rgb_trajectory.py
git commit -m "feat: render RGB trajectory MP4"
```

### Task 3: Add the Demo 10 CLI

**Files:**

- Create: `demo/10_rgb_trajectory.py`
- Modify: `tests/test_rgb_trajectory.py`

**Interfaces:**

- Consumes `load_rgb_trajectory`, `export_rgb_ply_sequence`, and `export_rgb_trajectory_mp4` from Tasks 1–2.
- Produces `parse_args(argv: list[str] | None = None) -> argparse.Namespace` and `main(argv: list[str] | None = None) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_demo_defaults_to_24_fps(self):
    demo = load_demo_10()

    args = demo.parse_args([
        "--dataset-root", "data/train", "--sample-id", "sample_0",
        "--object-id", "000", "--output-dir", "outputs/trajectory",
    ])

    self.assertEqual(args.fps, 24)


def test_demo_rejects_nonpositive_fps(self):
    demo = load_demo_10()

    with self.assertRaisesRegex(ValueError, "fps"):
        demo.main([
            "--dataset-root", "data/train", "--sample-id", "sample_0",
            "--object-id", "000", "--output-dir", "outputs/trajectory", "--fps", "0",
        ])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n utonia-dev python -m unittest tests.test_rgb_trajectory -v`

Expected: FAIL because `demo/10_rgb_trajectory.py` does not exist.

- [ ] **Step 3: Implement the thin CLI**

```python
def main(argv=None):
    args = parse_args(argv)
    if args.fps < 1:
        raise ValueError("fps must be a positive integer")
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
```

- [ ] **Step 4: Run focused verification**

Run: `conda run -n utonia-dev python -m unittest tests.test_rgb_trajectory tests.test_joint_trajectory_pca -v && python -m py_compile demo/10_rgb_trajectory.py utonia/rgb_trajectory.py && git diff --check`

Expected: PASS with no syntax or whitespace errors.

- [ ] **Step 5: Commit the CLI and tests**

```bash
git add demo/10_rgb_trajectory.py tests/test_rgb_trajectory.py
git commit -m "feat: add RGB trajectory export demo"
```
