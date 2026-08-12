# Inline Utonia Container Dependencies Design

## Goal

Make `utonia.def` self-contained like `edited-physctrl/pc.def`, so it builds
the Utonia runtime without copying or reading `environment.yml`.

## Selected approach

Use the CUDA 12.6.3 Ubuntu 22.04 base image and keep the Miniforge installation. In
`%post`, create `/opt/conda/envs/utonia` with Python 3.10 plus build tooling,
then install every Utonia dependency explicitly:

- Set `CONDA_OVERRIDE_CUDA=12.6` and install PyTorch 2.4.1, torchvision
  0.19.1, and torchaudio 2.4.1 from the CUDA 12.4 PyTorch wheel index. This
  Torch ABI matches the requested FlashAttention wheel.
- Install `torch-scatter` from the PyG PyTorch 2.4/CUDA 12.4 wheel index and
  install `spconv-cu124`; CUDA 12.x compatibility permits these extension
  wheels on the CUDA 12.6 base.
- Install the requested precompiled FlashAttention wheel after the compatible
  PyTorch stack is present:
  `https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.6.3+cu126torch2.4-cp310-cp310-linux_x86_64.whl`.
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
does not reference `environment.yml`, that it uses CUDA 12.6.3, the Torch 2.4
stack, and the exact requested precompiled FlashAttention wheel, and that the
existing bind-mount `%help` command remains intact. Build and GPU import
validation will run on the remote Apptainer cluster.
