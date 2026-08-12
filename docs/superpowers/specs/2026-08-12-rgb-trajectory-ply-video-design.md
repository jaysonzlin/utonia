# RGB Trajectory PLY and Video Design

## Goal

Add Demo 10, a GPU-independent command that exports every frame of one
NGFF-derived object trajectory as an RGB PLY sequence and produces a stable,
VS Code-playable MP4 preview of that sequence.

## Command-line interface

`demo/10_rgb_trajectory.py` accepts:

- `--dataset-root`: directory containing `sample_*` directories.
- `--sample-id`: sample directory name, such as `sample_0`.
- `--object-id`: object HDF5 directory name, such as `000`.
- `--output-dir`: directory that receives the PLY sequence and MP4.
- `--fps`: optional positive integer; defaults to `24`.

The command needs neither CUDA nor a Utonia checkpoint.

## Inputs and validation

The selected input is
`<dataset-root>/<sample-id>/objects/<object-id>/pc.hdf5`. The HDF5 file must
contain:

- `point_cloud` with shape `[T, 1, N, 3]` and numeric coordinates.
- `rgb` with shape `[N, 3]`, numeric finite values, and every value in
  `[0, 1]`.

The command rejects path-like sample and object IDs, missing files or
datasets, invalid shapes, non-numeric values, invalid RGB ranges, and invalid
FPS values with contextual error messages.

## Outputs

For trajectory frame `t`, Demo 10 writes an ASCII PLY named
`<sample-id>_object_<object-id>_frame_<t:04d>.ply`. Every PLY contains the
frame's original coordinates plus static RGB converted to `uchar red`,
`green`, and `blue` fields.

It writes one MP4 named `<sample-id>_object_<object-id>_trajectory.mp4`. The
MP4 contains one rendered frame per trajectory frame and uses the requested
FPS.

## Rendering

Matplotlib's non-interactive Agg backend renders the RGB points. OpenCV's
`VideoWriter` encodes the raster frames as MP4. The renderer computes a single
world-space bound over all trajectory frames, chooses one fixed azimuth and
elevation, and reuses the same axis limits and aspect ratio for every video
frame. This preserves motion without camera jitter.

The video uses a white background, no depth shading, and a point size chosen
for the 2,048-point object clouds. The title identifies the sample, object,
and zero-based trajectory frame.

## Structure and tests

Trajectory HDF5 parsing and PLY serialization belong in a new focused helper
module under `utonia/`, so Demo 10 remains a thin CLI. Unit tests use temporary
HDF5 inputs to verify sorted frame export, byte-level PLY color conversion,
fixed bounds, RGB validation, and CLI argument defaults. The MP4 writer is
tested with a short real temporary video and verified as non-empty.

## Dependencies

The implementation relies on existing Utonia demo dependencies: `h5py`,
NumPy, Matplotlib, and OpenCV. It does not add dependencies or modify
`environment.yml` or `utonia.def`.
