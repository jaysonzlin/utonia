# Joint-Trajectory PCA Visualization Design

## Goal

Provide a CUDA-only Utonia CLI that reads a sample produced for
`train_joint_wan_physctrl_trajectory.py` and writes one PCA-colored PLY file
for each object at the trainer's conditioning frame.

## Context

The joint trajectory trainer reads each object from
`<dataset-root>/<sample-id>/objects/<object-id>/pc.hdf5`.  Its
`point_cloud` dataset has shape `[T, 1, N, 3]`.  For a trajectory start frame
`s` (one-based), the trainer uses `point_cloud[s - 1, 0]` as the initial point
cloud.  The visualization must select that exact cloud.

## Considered approaches

1. Import `JointWanPhysCtrlDataset` from the Wan repository.  This would
   duplicate the trainer's loader, but makes Utonia depend on the Wan runtime
   and its video-validation requirements.
2. Export training clouds to an intermediate format first.  This adds an
   extra, error-prone preprocessing step and can make the visualized data
   diverge from the training data.
3. Read the documented HDF5 layout directly from Utonia.  This keeps the
   tool self-contained while selecting the identical coordinates used by the
   trainer.  This is the selected approach.

## CLI and data flow

Add `demo/9_pca_joint_trajectory.py` with these required arguments:

- `--dataset-root`: root containing training sample directories.
- `--sample-id`: selected training sample directory name.
- `--start-frame`: one-based trajectory conditioning frame.
- `--output-dir`: directory where exported PLY files are written.

Optional arguments:

- `--checkpoint`: local Utonia checkpoint.  If omitted, load the official
  `Pointcept/Utonia` checkpoint.
- `--seed`: deterministic seed, default `37`.

The CLI discovers object directories lexically, as the trainer does.  For each
object it validates that `pc.hdf5` contains `point_cloud`, checks its
`[T, 1, N, 3]` structure, checks `start_frame` is within `T`, and extracts
`point_cloud[start_frame - 1, 0]`.

Each object is encoded independently.  The input has zero-filled color and
normal arrays because the source training format contains coordinates only.
Use Utonia's default transform with `normalize_coord=True` and `scale=1.0`.
The transform therefore normalizes only the inference representation; the
export retains the source coordinates unchanged.

Load the model on CUDA only.  If CUDA is unavailable, fail before loading a
model with an actionable error.  If FlashAttention is unavailable, use
Utonia's existing CUDA-compatible non-FlashAttention model configuration.

After inference, upcast features through Utonia's pooling hierarchy and map
them through `inverse` so every source point gets a feature.  Independently
per object, apply the PCA-to-RGB mapping used by `demo/6_pca_object.py`:
`torch.pca_lowrank(..., q=min(9, N, D), niter=5)`, a 60/40 blend of components
1--3 and 4--6, then per-channel normalization and clamping.  Write those colors and
the original coordinates to `<output-dir>/<sample-id>_frame_<raw-frame>_<object-id>_pca.ply`.
No viewer is opened and no raw-color PLY is written.

## Structure

Keep the module testable by separating side-effect-free helpers from the CLI:

- `parse_args` owns command-line parsing.
- `load_initial_point_clouds` validates the dataset layout and returns ordered
  object IDs plus NumPy coordinate arrays.
- `build_object_input` produces Utonia's coordinate-only input dictionary.
- `load_utonia_model` enforces CUDA and handles official/local checkpoints.
- `upcast_features` restores Utonia features to the transformed cloud.
- `pca_colors` derives RGB values from one object's features.
- `export_object_pca` maps colors to source points and writes a PLY.
- `main` coordinates those helpers without opening a viewer.

## Error handling

Fail with clear `ValueError` or `FileNotFoundError` messages for a missing
sample, missing `objects` directory, missing/invalid HDF5 data, no object
directories, or an out-of-range start frame.  Fail with `RuntimeError` when
CUDA is unavailable.  Do not silently default the frame, sample, or output
location.

## Testing

Add unit tests that create minimal temporary HDF5 sample directories and
verify object ordering, selection of `start_frame - 1`, invalid-shape and
out-of-range rejection, zero modality construction, and PCA-color bounds.
Tests must not download a checkpoint, use CUDA, or open a viewer.  A final
smoke check will compile the new demo and exercise its `--help` command.

## Documentation

Add a concise README example showing the command, including a `--start-frame`
value copied from the active trajectory configuration, and explain that each
PLY contains original coordinates with independently-derived Utonia PCA RGB
colors.
