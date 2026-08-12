# PCA Trajectory Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Demo 11 to export independently PCA-colored Utonia trajectory PLYs and a fixed-camera MP4 for one selected object.

**Architecture:** Extend the Demo 10 renderer with a public color-agnostic MP4 function while retaining its RGB wrapper. Put frame-wise Utonia encoding and PCA artifact export in `utonia/pca_trajectory.py`; `demo/11_pca_trajectory.py` will only parse arguments, enforce CUDA, load the model, and orchestrate the helper.

**Tech Stack:** Python 3.10, PyTorch CUDA, NumPy, h5py, Matplotlib Agg, OpenCV VideoWriter, unittest.

## Global Constraints

- Do not modify `environment.yml` or `utonia.def`.
- Require CUDA and reject non-CUDA hosts before importing Utonia.
- Consume one selected object from `point_cloud [T, 1, N, 3]` and static `rgb [N, 3]` using `load_rgb_trajectory`.
- Feed each frame's original coordinates and static HDF5 RGB to Utonia, with zero normals and `utonia.transform.default(scale=1.0, normalize_coord=True)`.
- Recompute `pca_colors` independently for every trajectory frame.
- Write `<sample-id>_object_<object-id>_frame_<t:04d>_pca.ply` with original coordinates and PCA RGB.
- Write `<sample-id>_object_<object-id>_pca_trajectory.mp4` at a default of 24 FPS, with positive `--fps` override.
- Use a fixed camera and padded full-trajectory bounds for every MP4 frame.

---

## File structure

- Modify `utonia/rgb_trajectory.py`: expose a reusable fixed-camera MP4 exporter that accepts any per-frame colors while retaining the existing RGB API.
- Create `utonia/pca_trajectory.py`: frame-wise Utonia encoding, independent PCA colors, PCA PLY sequence, and PCA MP4 orchestration.
- Create `demo/11_pca_trajectory.py`: CUDA-gated Demo 11 CLI.
- Create `tests/test_pca_trajectory.py`: fake-model PCA orchestration and CLI behavior tests.
- Modify `tests/test_rgb_trajectory.py`: test the color-agnostic renderer API and retain the Demo 10 regression test.

### Task 1: Generalize Demo 10's fixed-camera renderer

**Files:**

- Modify: `utonia/rgb_trajectory.py`
- Modify: `tests/test_rgb_trajectory.py`

**Interfaces:**

- Produces `export_colored_trajectory_mp4(output_dir: Path | str, filename: str, title_prefix: str, trajectory: np.ndarray, colors: np.ndarray, fps: int) -> Path`.
- Keeps `export_rgb_trajectory_mp4(...) -> Path` as a wrapper using static `[N, 3]` RGB and the existing filename.
- Consumes `trajectory [T, N, 3]`, either static colors `[N, 3]` or frame-varying colors `[T, N, 3]`, and a positive FPS.

- [ ] **Step 1: Write the failing tests**

```python
def test_colored_renderer_writes_frame_varying_colors_to_mp4(self):
    trajectory = np.stack([frame_points(0), frame_points(1)])
    colors = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
    ], dtype=np.float32)

    path = rgb_trajectory.export_colored_trajectory_mp4(
        self.root, "sample_0_object_000_pca_trajectory.mp4", "sample_0 · object 000",
        trajectory, colors, fps=24,
    )

    self.assertEqual(path.name, "sample_0_object_000_pca_trajectory.mp4")
    self.assertGreater(path.stat().st_size, 0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `MPLCONFIGDIR=/private/tmp/mplconfig conda run -n utonia-dev python -m unittest tests.test_rgb_trajectory.RgbTrajectoryTests.test_colored_renderer_writes_frame_varying_colors_to_mp4 -v`

Expected: FAIL because `export_colored_trajectory_mp4` does not exist.

- [ ] **Step 3: Implement the color-agnostic exporter**

```python
def export_colored_trajectory_mp4(output_dir, filename, title_prefix, trajectory, colors, fps):
    trajectory = _validate_trajectory(trajectory)
    colors = _validate_frame_colors(trajectory, colors)
    lower, upper = trajectory_bounds(trajectory)
    # Render frame i with colors[i], fixed bounds, elevation=20, azimuth=-55,
    # then append BGR raster data to an mp4v cv2.VideoWriter.


def export_rgb_trajectory_mp4(output_dir, sample_id, object_id, trajectory, rgb, fps):
    trajectory, rgb = _validate_trajectory_and_rgb(trajectory, rgb)
    return export_colored_trajectory_mp4(
        output_dir,
        f"{sample_id}_object_{object_id}_trajectory.mp4",
        f"{sample_id} · object {object_id}",
        trajectory,
        np.broadcast_to(rgb, trajectory.shape),
        fps,
    )
```

- [ ] **Step 4: Run the renderer tests to verify they pass**

Run: `MPLCONFIGDIR=/private/tmp/mplconfig conda run -n utonia-dev python -m unittest tests.test_rgb_trajectory -v`

Expected: PASS, including the existing static-RGB MP4 regression test.

- [ ] **Step 5: Commit the renderer refactor**

```bash
git add utonia/rgb_trajectory.py tests/test_rgb_trajectory.py
git commit -m "refactor: share trajectory video renderer"
```

### Task 2: Build frame-wise PCA trajectory helpers

**Files:**

- Create: `utonia/pca_trajectory.py`
- Create: `tests/test_pca_trajectory.py`

**Interfaces:**

- Consumes `load_rgb_trajectory` and `export_colored_trajectory_mp4` from Task 1, plus `build_object_input`, `move_tensors_to_cuda`, `upcast_features`, and `pca_colors` from `utonia.joint_trajectory_pca`.
- Produces `encode_pca_trajectory(trajectory: np.ndarray, rgb: np.ndarray, transform: Callable, model: Callable, move_to_cuda: Callable) -> np.ndarray` with frame-wise PCA colors `[T, N, 3]`.
- Produces `export_pca_ply_sequence(output_dir: Path | str, sample_id: str, object_id: str, trajectory: np.ndarray, pca_colors_by_frame: np.ndarray) -> list[Path]`.
- Produces `export_pca_trajectory_mp4(output_dir: Path | str, sample_id: str, object_id: str, trajectory: np.ndarray, pca_colors_by_frame: np.ndarray, fps: int) -> Path`.

- [ ] **Step 1: Write the failing tests**

```python
def test_encodes_each_frame_with_static_rgb_and_independent_pca(self):
    trajectory = np.stack([frame_points(0), frame_points(1)])
    rgb = np.full((2, 3), 0.25, dtype=np.float32)
    captured_colors = []

    def transform(point):
        captured_colors.append(point["color"].copy())
        return FakePoint(torch.arange(12, dtype=torch.float32).reshape(2, 6))

    with patch.object(pca_trajectory, "pca_colors", side_effect=[
        torch.zeros((2, 3)), torch.ones((2, 3)),
    ]) as mapping:
        colors = pca_trajectory.encode_pca_trajectory(
            trajectory, rgb, transform, lambda point: point, lambda point: point
        )

    self.assertEqual(mapping.call_count, 2)
    np.testing.assert_array_equal(captured_colors[0], rgb)
    np.testing.assert_array_equal(captured_colors[1], rgb)
    np.testing.assert_array_equal(colors[0], np.zeros((2, 3)))
    np.testing.assert_array_equal(colors[1], np.ones((2, 3)))


def test_pca_ply_sequence_and_mp4_have_pca_names(self):
    colors = np.stack([np.zeros((2, 3)), np.ones((2, 3))]).astype(np.float32)

    plys = pca_trajectory.export_pca_ply_sequence(
        self.root, "sample_0", "000", self.trajectory, colors
    )
    mp4 = pca_trajectory.export_pca_trajectory_mp4(
        self.root, "sample_0", "000", self.trajectory, colors, 24
    )

    self.assertEqual(plys[0].name, "sample_0_object_000_frame_0000_pca.ply")
    self.assertEqual(mp4.name, "sample_0_object_000_pca_trajectory.mp4")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `MPLCONFIGDIR=/private/tmp/mplconfig conda run -n utonia-dev python -m unittest tests.test_pca_trajectory -v`

Expected: FAIL because `utonia/pca_trajectory.py` does not exist.

- [ ] **Step 3: Implement frame-wise PCA behavior**

```python
def encode_pca_trajectory(trajectory, rgb, transform, model, move_to_cuda):
    trajectory, rgb = _validate_trajectory_and_rgb(trajectory, rgb)
    colors_by_frame = []
    for coords in trajectory:
        point = move_to_cuda(transform(build_object_input(coords, rgb)))
        with torch.inference_mode():
            point = upcast_features(model(point))
        colors_by_frame.append(pca_colors(point.feat)[point.inverse].cpu().numpy())
    return np.stack(colors_by_frame).astype(np.float32)
```

Implement the PLY writer with `x`, `y`, `z`, `uchar red`, `uchar green`, and
`uchar blue` properties, then call Task 1's generic renderer with the PCA
filename and title prefix.

- [ ] **Step 4: Run the helper tests to verify they pass**

Run: `MPLCONFIGDIR=/private/tmp/mplconfig conda run -n utonia-dev python -m unittest tests.test_pca_trajectory -v`

Expected: PASS, including one real non-empty MP4 output.

- [ ] **Step 5: Commit PCA helpers and tests**

```bash
git add utonia/pca_trajectory.py tests/test_pca_trajectory.py
git commit -m "feat: export frame-wise PCA trajectory artifacts"
```

### Task 3: Add Demo 11 CLI and CUDA gate

**Files:**

- Create: `demo/11_pca_trajectory.py`
- Modify: `tests/test_pca_trajectory.py`

**Interfaces:**

- Consumes `load_rgb_trajectory`, `encode_pca_trajectory`, `export_pca_ply_sequence`, `export_pca_trajectory_mp4`, `load_utonia_model`, and `require_cuda`.
- Produces `parse_args(argv: list[str] | None = None) -> argparse.Namespace` and `main(argv: list[str] | None = None) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_demo_defaults_to_demo_9_model_selection_and_24_fps(self):
    demo = load_demo_11()

    args = demo.parse_args([
        "--dataset-root", "data/train", "--sample-id", "sample_0",
        "--object-id", "000", "--output-dir", "outputs/pca",
    ])

    self.assertEqual(args.fps, 24)
    self.assertEqual(args.seed, 37)
    self.assertIsNone(args.checkpoint)


@patch("torch.cuda.is_available", return_value=False)
def test_demo_rejects_non_cuda_host_before_importing_utonia(self, _available):
    demo = load_demo_11()

    with self.assertRaisesRegex(RuntimeError, "CUDA"):
        demo.main([
            "--dataset-root", "data/train", "--sample-id", "sample_0",
            "--object-id", "000", "--output-dir", "outputs/pca",
        ])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n utonia-dev python -m unittest tests.test_pca_trajectory -v`

Expected: FAIL because `demo/11_pca_trajectory.py` does not exist.

- [ ] **Step 3: Implement the thin CUDA-only CLI**

```python
def main(argv=None):
    args = parse_args(argv)
    if args.fps < 1:
        raise ValueError("fps must be a positive integer")
    require_cuda()
    import utonia
    utonia.utils.set_seed(args.seed)
    model = load_utonia_model(args.checkpoint)
    transform = utonia.transform.default(scale=1.0, normalize_coord=True)
    trajectory, rgb = load_rgb_trajectory(args.dataset_root, args.sample_id, args.object_id)
    colors = encode_pca_trajectory(trajectory, rgb, transform, model, move_tensors_to_cuda)
    plys = export_pca_ply_sequence(args.output_dir, args.sample_id, args.object_id, trajectory, colors)
    mp4 = export_pca_trajectory_mp4(args.output_dir, args.sample_id, args.object_id, trajectory, colors, args.fps)
    for path in [*plys, mp4]:
        print(path)
```

- [ ] **Step 4: Run full focused verification**

Run: `MPLCONFIGDIR=/private/tmp/mplconfig conda run -n utonia-dev python -m unittest discover -s tests -v && python -m py_compile demo/11_pca_trajectory.py utonia/pca_trajectory.py utonia/rgb_trajectory.py && git diff --check`

Expected: PASS with no syntax or whitespace errors.

- [ ] **Step 5: Commit Demo 11 and tests**

```bash
git add demo/11_pca_trajectory.py tests/test_pca_trajectory.py
git commit -m "feat: add PCA trajectory video demo"
```
