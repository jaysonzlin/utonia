# Joint-Trajectory PCA Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CUDA-only Utonia command that exports an independently PCA-colored PLY for every object in a selected joint-training sample at the exact trajectory conditioning frame.

**Architecture:** Put data-layout validation, coordinate-only input creation, PCA coloring, and PLY export in `utonia/joint_trajectory_pca.py`. Keep `demo/9_pca_joint_trajectory.py` as a thin CLI that invokes those helpers sequentially for every object.

**Tech Stack:** Python 3.10, NumPy, PyTorch, h5py, standard ASCII PLY, Utonia transforms/model loader, standard-library unittest.

## Global Constraints

- Read only `<dataset-root>/<sample-id>/objects/<object-id>/pc.hdf5`, dataset `point_cloud` shaped `[T, 1, N, 3]`.
- Require one-based `--start-frame`; select raw index `start_frame - 1` and never default it.
- Process objects independently with zero color/normal inputs, `normalize_coord=True`, and Utonia scale `1.0`.
- Preserve original source coordinates in PLYs and use only Utonia PCA features for RGB.
- Require CUDA and never open a viewer.
- Default to `Pointcept/Utonia` and accept a local `--checkpoint`.
- Seed Utonia with default `37`; use the existing object demo's six-component weighted PCA map independently per object.

---

### Task 1: Add testable training-data selection

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_joint_trajectory_pca.py`
- Create: `utonia/joint_trajectory_pca.py`

**Interfaces:**
- Consumes: `dataset_root: Path`, `sample_id: str`, `start_frame: int`.
- Produces: `load_initial_point_clouds(dataset_root, sample_id, start_frame) -> list[tuple[str, np.ndarray]]`, sorted lexically by object ID with each array shaped `[N, 3]`.

- [ ] **Step 1: Write the failing loader tests**

```python
def test_loads_sorted_objects_at_the_requested_condition_frame(self):
    write_cloud(self.root, "sample-a", "object-b", frames=3, offset=100)
    write_cloud(self.root, "sample-a", "object-a", frames=3, offset=0)

    loaded = load_initial_point_clouds(self.root, "sample-a", start_frame=2)

    self.assertEqual([name for name, _ in loaded], ["object-a", "object-b"])
    np.testing.assert_array_equal(loaded[0][1], frame_points(0, 1))
    np.testing.assert_array_equal(loaded[1][1], frame_points(100, 1))

def test_rejects_out_of_range_start_frame(self):
    write_cloud(self.root, "sample-a", "object-a", frames=2, offset=0)

    with self.assertRaisesRegex(ValueError, "start frame 3.*2"):
        load_initial_point_clouds(self.root, "sample-a", start_frame=3)

def test_rejects_non_training_point_cloud_shape(self):
    write_invalid_cloud(self.root, "sample-a", "object-a", shape=(2, 4, 3))

    with self.assertRaisesRegex(ValueError, "shape [T, 1, N, 3]"):
        load_initial_point_clouds(self.root, "sample-a", start_frame=1)
```

The temporary-test helpers must create the exact `objects/<object-id>/pc.hdf5` hierarchy with deterministic float32 `point_cloud` data.

- [ ] **Step 2: Verify the test fails**

Run: `python -m unittest tests.test_joint_trajectory_pca.LoadInitialPointCloudsTests -v`

Expected: FAIL because `utonia.joint_trajectory_pca` does not exist.

- [ ] **Step 3: Implement the minimal HDF5 loader**

```python
def load_initial_point_clouds(dataset_root: Path, sample_id: str, start_frame: int):
    if start_frame < 1:
        raise ValueError("start_frame must be one-based and at least 1")
    object_root = Path(dataset_root) / sample_id / "objects"
    if not object_root.is_dir():
        raise FileNotFoundError(f"missing objects directory: {object_root}")
    object_dirs = sorted(path for path in object_root.iterdir() if path.is_dir())
    if not object_dirs:
        raise ValueError(f"no object directories found in {object_root}")
    # Require pc.hdf5/point_cloud, validate [T, 1, N, 3], and return
    # np.asarray(cloud[start_frame - 1, 0], dtype=np.float32) per object.
```

- [ ] **Step 4: Verify the loader tests pass**

Run: `python -m unittest tests.test_joint_trajectory_pca.LoadInitialPointCloudsTests -v`

Expected: PASS for all three tests.

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/test_joint_trajectory_pca.py utonia/joint_trajectory_pca.py
git commit -m "feat: load joint trajectory condition clouds"
```

### Task 2: Add coordinate-only inputs and PCA RGB helpers

**Files:**
- Modify: `utonia/joint_trajectory_pca.py`
- Modify: `tests/test_joint_trajectory_pca.py`

**Interfaces:**
- Consumes: `coords: np.ndarray` shaped `[N, 3]` and `features: torch.Tensor` shaped `[N, D]`, `D >= 6`.
- Produces: `build_object_input(coords) -> dict[str, np.ndarray]` and `pca_colors(features, brightness=1.2) -> torch.Tensor` shaped `[N, 3]` within `[0, 1]`.

- [ ] **Step 1: Write the failing helper tests**

```python
def test_build_object_input_zero_fills_missing_modalities(self):
    coords = np.arange(12, dtype=np.float32).reshape(4, 3)

    point = build_object_input(coords)

    np.testing.assert_array_equal(point["coord"], coords)
    np.testing.assert_array_equal(point["color"], np.zeros_like(coords))
    np.testing.assert_array_equal(point["normal"], np.zeros_like(coords))

def test_pca_colors_returns_bounded_rgb_for_one_object(self):
    features = torch.arange(96, dtype=torch.float32).reshape(12, 8)

    colors = pca_colors(features)

    self.assertEqual(tuple(colors.shape), (12, 3))
    self.assertTrue(torch.all((0 <= colors) & (colors <= 1)))
```

- [ ] **Step 2: Verify the test fails**

Run: `python -m unittest tests.test_joint_trajectory_pca.JointTrajectoryPcaHelperTests -v`

Expected: FAIL because the two helpers are absent.

- [ ] **Step 3: Implement minimal helpers**

```python
def build_object_input(coords):
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape [N, 3]")
    return {"coord": coords.copy(), "color": np.zeros_like(coords), "normal": np.zeros_like(coords)}

def pca_colors(features, brightness=1.2):
    if features.ndim != 2 or features.shape[1] < 6:
        raise ValueError("features must have shape [N, D] with D >= 6")
    _, _, basis = torch.pca_lowrank(features, center=True, niter=5, q=min(9, *features.shape))
    projection = features @ basis
    projection = projection[:, :3] * 0.6 + projection[:, 3:6] * 0.4
    minimum = projection.min(dim=0, keepdim=True).values
    maximum = projection.max(dim=0, keepdim=True).values
    return ((projection - minimum) / (maximum - minimum).clamp_min(1e-6) * brightness).clamp(0, 1)
```

- [ ] **Step 4: Verify the tests pass**

Run: `python -m unittest tests.test_joint_trajectory_pca.JointTrajectoryPcaHelperTests -v`

Expected: PASS for both tests.

- [ ] **Step 5: Commit**

```bash
git add utonia/joint_trajectory_pca.py tests/test_joint_trajectory_pca.py
git commit -m "feat: prepare and color joint trajectory objects"
```

### Task 3: Add CUDA inference helpers and PLY export

**Files:**
- Modify: `utonia/joint_trajectory_pca.py`
- Modify: `tests/test_joint_trajectory_pca.py`

**Interfaces:**
- Produces: `require_cuda() -> str`, `load_utonia_model(checkpoint: str | None)`, `upcast_features(point)`, `move_tensors_to_cuda(point)`, and `export_object_pca(...) -> Path`.

- [ ] **Step 1: Write the failing CUDA/export tests**

```python
@patch("utonia.joint_trajectory_pca.torch.cuda.is_available", return_value=False)
def test_require_cuda_fails_with_an_actionable_message(self, _available):
    with self.assertRaisesRegex(RuntimeError, "CUDA"):
        require_cuda()

def test_export_writes_original_coordinates_and_pca_rgb(self):
    coords = np.arange(12, dtype=np.float32).reshape(4, 3)
    colors = np.full((4, 3), 0.5, dtype=np.float32)

    path = export_object_pca(self.root, "sample-a", 12, "object-a", coords, colors)

    self.assertEqual(path.name, "sample-a_frame_0012_object-a_pca.ply")
    lines = path.read_text().splitlines()
    self.assertEqual(lines[0], "ply")
    self.assertEqual(lines[10], "0.0 1.0 2.0 128 128 128")
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m unittest tests.test_joint_trajectory_pca.JointTrajectoryPcaExportTests -v`

Expected: FAIL because the interfaces are absent.

- [ ] **Step 3: Implement the selected runtime behavior**

```python
def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run joint trajectory PCA visualization")
    return "cuda"

def export_object_pca(output_dir, sample_id, raw_frame, object_id, coords, colors):
    path = Path(output_dir) / f"{sample_id}_frame_{raw_frame:04d}_{object_id}_pca.ply"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write an ASCII PLY header followed by the original coordinates and
    # clamped RGB values converted from [0, 1] to unsigned 8-bit channels.
    return path
```

Implement `load_utonia_model` using `require_cuda()`, official or local weights, `.to("cuda")`, `.eval()`, and the existing non-FlashAttention fallback (`enc_patch_size=[1024] * 5`, `enable_flash=False`). Implement `upcast_features` with the exact hierarchy traversal in `demo/6_pca_object.py`, retaining `inverse` for the final source-point color mapping.

- [ ] **Step 4: Verify export tests pass**

Run: `python -m unittest tests.test_joint_trajectory_pca.JointTrajectoryPcaExportTests -v`

Expected: PASS with no CUDA usage, weight download, Open3D dependency, or viewer.

- [ ] **Step 5: Commit**

```bash
git add utonia/joint_trajectory_pca.py tests/test_joint_trajectory_pca.py
git commit -m "feat: export trajectory PCA point clouds"
```

### Task 4: Add CLI, dependency, and usage documentation

**Files:**
- Create: `demo/9_pca_joint_trajectory.py`
- Modify: `environment.yml`
- Modify: `README.md`
- Modify: `tests/test_joint_trajectory_pca.py`

**Interfaces:**
- Consumes: required `--dataset-root`, `--sample-id`, `--start-frame`, and `--output-dir`; optional `--checkpoint` and `--seed`.
- Produces: one `<sample-id>_frame_<raw-frame>_<object-id>_pca.ply` file and printed path per lexical object directory.

- [ ] **Step 1: Write the failing parser test**

```python
def test_parse_args_requires_training_dataset_identifiers(self):
    args = parse_args([
        "--dataset-root", "data/train",
        "--sample-id", "000001",
        "--start-frame", "13",
        "--output-dir", "outputs/pca",
    ])

    self.assertEqual(args.start_frame, 13)
    self.assertEqual(args.seed, 37)
    self.assertIsNone(args.checkpoint)
```

Load the numeric demo filename with `importlib.util.spec_from_file_location` in the test.

- [ ] **Step 2: Verify the parser test fails**

Run: `python -m unittest tests.test_joint_trajectory_pca.JointTrajectoryPcaCliTests -v`

Expected: FAIL because `demo/9_pca_joint_trajectory.py` is absent.

- [ ] **Step 3: Implement the thin CLI and docs**

```python
def main(argv=None):
    args = parse_args(argv)
    utonia.utils.set_seed(args.seed)
    model = load_utonia_model(args.checkpoint)
    transform = utonia.transform.default(scale=1.0, normalize_coord=True)
    for object_id, coords in load_initial_point_clouds(args.dataset_root, args.sample_id, args.start_frame):
        point = move_tensors_to_cuda(transform(build_object_input(coords)))
        with torch.inference_mode():
            point = upcast_features(model(point))
        colors = pca_colors(point.feat)[point.inverse].cpu().numpy()
        print(export_object_pca(args.output_dir, args.sample_id, args.start_frame - 1, object_id, coords, colors))
```

Add `h5py` to the conda dependencies. Add this README command in the visualization section and state that it is CUDA-only, opens no viewer, preserves original coordinates, and colors each object with its own PCA basis:

```bash
export PYTHONPATH=./
python demo/9_pca_joint_trajectory.py \
  --dataset-root /path/to/td_832x480_3_soft \
  --sample-id YOUR_SAMPLE_ID \
  --start-frame 13 \
  --output-dir outputs/joint_trajectory_pca
```

- [ ] **Step 4: Verify unit tests and static CLI behavior**

Run: `python -m unittest discover -s tests -v && python -m py_compile utonia/joint_trajectory_pca.py demo/9_pca_joint_trajectory.py && python demo/9_pca_joint_trajectory.py --help`

Expected: all tests pass, both files compile, and help lists required arguments without checking CUDA or loading a model.

- [ ] **Step 5: Commit**

```bash
git add demo/9_pca_joint_trajectory.py environment.yml README.md tests/test_joint_trajectory_pca.py
git commit -m "docs: describe joint trajectory PCA visualization"
```

### Task 5: Verify requirements and publish

**Files:**
- Verify: `docs/superpowers/specs/2026-08-12-joint-trajectory-pca-design.md`
- Verify: `docs/superpowers/plans/2026-08-12-joint-trajectory-pca.md`
- Verify: `utonia/joint_trajectory_pca.py`, `demo/9_pca_joint_trajectory.py`, `tests/test_joint_trajectory_pca.py`, `README.md`, and `environment.yml`.

**Interfaces:**
- Produces: fresh verification evidence before pushing to `origin/main`.

- [ ] **Step 1: Run fresh full verification**

Run: `python -m unittest discover -s tests -v && python -m py_compile utonia/joint_trajectory_pca.py demo/9_pca_joint_trajectory.py && python demo/9_pca_joint_trajectory.py --help && git diff --check && git status --short`

Expected: tests, compilation, and help succeed; whitespace is clean; status contains only intended files.

- [ ] **Step 2: Check design requirements line by line**

Confirm code selects `[start_frame - 1, 0]`; sorts object directories; supplies zero modalities; uses `scale=1.0, normalize_coord=True`; checks CUDA before model loading; computes PCA per object; exports original coordinates only to PLY; and documents the command and h5py dependency.

- [ ] **Step 3: Commit planning documents and final corrections**

```bash
git add docs/superpowers/specs/2026-08-12-joint-trajectory-pca-design.md docs/superpowers/plans/2026-08-12-joint-trajectory-pca.md
git commit -m "docs: plan joint trajectory PCA visualization"
```

- [ ] **Step 4: Verify and publish**

Run: `git remote -v && git log --oneline origin/main..main && git push origin main`

Expected: both remote URLs are `git@github.com:jaysonzlin/utonia.git`, local commits are listed, and the push updates `main`.
