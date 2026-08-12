# Inline Utonia Container Dependencies Design

## Goal

Make `utonia.def` self-contained like `edited-physctrl/pc.def`, so it builds
the Utonia runtime without copying or reading `environment.yml`.

## Selected approach

Keep the CUDA 12.4.1 Ubuntu 22.04 base image and Miniforge installation.  In
`%post`, create `/opt/conda/envs/utonia` with Python 3.10 plus build tooling,
then install every Utonia dependency explicitly:

- Install PyTorch 2.5.0, torchvision 0.20.0, and torchaudio 2.5.0 from the
  CUDA 12.4 PyTorch wheel index.
- Install `torch-scatter` from the PyG wheel index matching PyTorch 2.5.0 and
  CUDA 12.4, and install `spconv-cu124`.
- Build FlashAttention from its Git repository after the CUDA-compatible
  PyTorch wheel is present.
- Install Utonia's remaining Python runtime packages inline: NumPy 1.26.4,
  SciPy, Addict, timm, psutil, huggingface_hub, h5py, Open3D, matplotlib,
  OpenCV, camtools, trimesh, natsort, gradio, and einops.

The source checkout remains outside the image and is bind-mounted at runtime.
`environment.yml` remains in the repository for Conda users but is not read by
the definition file.

## Alternatives considered

1. Continue using `mamba env create -f environment.yml`. This keeps one source
   of truth but makes the image definition depend on a separately maintained
   file. Rejected at the user's request.
2. Put all dependencies in a Conda YAML block embedded in the definition.
   This still treats the environment as an external specification and is less
   consistent with `pc.def`.
3. Use explicit Mamba setup plus ordered pip installs. Selected because it
   matches `pc.def` and makes CUDA/PyTorch extension compatibility visible in
   the container definition.

## Validation

Static validation must confirm that `utonia.def` has no `%files` section and
does not reference `environment.yml`, that its inline dependencies include the
CUDA 12.4 PyTorch stack plus Utonia's extension/runtime packages, and that the
existing bind-mount `%help` command remains intact. Build and GPU import
validation will run on the remote Apptainer cluster.
