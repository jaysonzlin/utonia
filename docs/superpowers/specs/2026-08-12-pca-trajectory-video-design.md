# PCA Trajectory Video Design

## Goal

Add Demo 11, a CUDA-only Utonia visualization command that encodes every
frame of one RGB point-cloud trajectory, independently computes PCA colors for
each frame, exports PCA-colored PLY files, and produces a stable MP4 preview.

## Command-line interface

`demo/11_pca_trajectory.py` accepts:

- `--dataset-root`: directory containing `sample_*` directories.
- `--sample-id`: sample directory name, such as `sample_0`.
- `--object-id`: object HDF5 directory name, such as `000`.
- `--output-dir`: directory that receives the PLY sequence and MP4.
- `--fps`: optional positive integer; defaults to `24`.
- `--checkpoint`: optional local Utonia checkpoint; the default is
  `Pointcept/Utonia`.
- `--seed`: optional deterministic seed; defaults to `37`.

The command requires CUDA and must fail with an actionable CUDA error before
loading Utonia when CUDA is unavailable.

## Inputs

The selected input is
`<dataset-root>/<sample-id>/objects/<object-id>/pc.hdf5`. It uses the existing
Demo 10 trajectory contract: numeric `point_cloud` `[T, 1, N, 3]` and static
finite `rgb` `[N, 3]` values in `[0, 1]`.

For every frame, Demo 11 passes the frame coordinates, static HDF5 RGB, and
zero normals through Utonia's default transform with scale `1.0` and normalized
coordinates. It runs Utonia encoding and restores point-level features using
the existing pooling upcast routine.

## PCA colors

Demo 11 invokes the existing PCA color mapping separately for each frame. Each
frame fits its own PCA basis and own RGB normalization from that frame's
encoded point features. Color flicker between frames is therefore expected and
intentional; it represents the requested independent frame-wise PCA.

## Outputs

For trajectory frame `t`, Demo 11 writes an ASCII PLY named
`<sample-id>_object_<object-id>_frame_<t:04d>_pca.ply`. The PLY uses original
frame coordinates and 8-bit colors derived from that frame's Utonia PCA
features.

It also writes `<sample-id>_object_<object-id>_pca_trajectory.mp4`, at the
requested FPS. The video contains one PCA-colored render per trajectory frame.

## Rendering

The MP4 uses the existing headless Matplotlib/OpenCV rendering approach from
Demo 10, with one padded world-space bound calculated from all coordinate
frames, a fixed elevation and azimuth, and identical limits for every frame.
Only PCA colors vary from frame to frame.

## Structure and tests

Shared PCA trajectory validation, per-frame encoding orchestration, and output
naming live in a focused `utonia/` helper. Demo 11 is a thin parser and CUDA
gate. Tests use controlled temporary HDF5 trajectories and fake model/transform
boundaries to verify RGB reaches the encoder, PCA is invoked once per frame,
each frame produces a PCA PLY, the MP4 is non-empty, CLI defaults are correct,
and non-CUDA hosts reject before Utonia import.

## Dependencies

This feature uses the existing Utonia, h5py, NumPy, Matplotlib, and OpenCV
dependencies. It does not modify `environment.yml` or `utonia.def`.
