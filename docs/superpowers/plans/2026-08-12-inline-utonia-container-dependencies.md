# Inline Utonia Container Dependencies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `utonia.def` install the full Utonia CUDA runtime inline, without copying or reading `environment.yml`.

**Architecture:** Use a CUDA 12.6.3 base while retaining the Miniforge root, GPU-driver cleanup, runtime environment variables, and bind-mount help. Keep the explicit Python 3.10 Mamba environment and ordered pip installation: Torch 2.4.1 first, matching extensions second, then the requested FlashAttention binary wheel and the remaining Utonia packages.

**Tech Stack:** Apptainer/Singularity definition syntax, Ubuntu 22.04, CUDA 12.6.3, Miniforge/Mamba, Python 3.10, PyTorch 2.4.1 CUDA 12.4 wheels, FlashAttention 2.6.3 CUDA 12.6/Torch 2.4 binary wheel.

## Global Constraints

- Do not include a `%files` section in `utonia.def`.
- Do not reference `environment.yml` anywhere in `utonia.def`.
- Keep source code external to the image and preserve the existing `--bind "$PWD:/workspace"` help recipe.
- Use CUDA 12.6.3 and set `CONDA_OVERRIDE_CUDA=12.6`.
- Pin PyTorch 2.4.1, torchvision 0.19.1, and torchaudio 2.4.1 to the CUDA 12.4 PyTorch index.
- Install `torch-scatter` from the `torch-2.4.0+cu124` PyG wheel index and install `spconv-cu124`.
- Install the exact requested FlashAttention precompiled wheel after PyTorch is installed; do not build it from Git.
- Preserve the NVIDIA host-driver cleanup and `--nv` runtime behavior.

---

### Task 1: Inline the dependency installation in the definition

**Files:**
- Modify: `utonia.def:1-71`

**Interfaces:**
- Consumes: public package indexes and the CUDA 12.4.1 base image during `apptainer build`.
- Produces: `/opt/conda/envs/utonia`, containing the complete Utonia Python runtime without requiring any repository file other than `utonia.def`.

- [ ] **Step 1: Establish the static failure condition**

Run: `rg -n '^%files|environment\.yml|mamba env create' utonia.def`

Expected: the current output shows the external-file dependency through `%files` and `mamba env create ... environment.yml`.

- [ ] **Step 2: Replace the external environment with explicit installation**

Delete:

```def
%files
    environment.yml /tmp/environment.yml
```

Replace the external environment creation with:

```def
ENV_BIN=/opt/conda/envs/utonia/bin
ENV_PATH=/opt/conda/envs/utonia
/opt/conda/bin/mamba create -y -p $ENV_PATH python=3.10 packaging ninja
$ENV_BIN/pip install --upgrade pip setuptools wheel
$ENV_BIN/pip install numpy==1.26.4 torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
$ENV_BIN/pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
$ENV_BIN/pip install spconv-cu124
$ENV_BIN/pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.6.3+cu126torch2.4-cp310-cp310-linux_x86_64.whl
```

Before FlashAttention, set `CPATH=$ENV_PATH/include:$CPATH`, `LIBRARY_PATH=$ENV_PATH/lib:$LIBRARY_PATH`, and `FORCE_CUDA=1`. Install the remaining packages in an inline `/tmp/requirements.txt` containing:

```text
scipy
addict
timm
psutil
huggingface_hub
h5py
open3d
matplotlib
opencv-python
camtools
trimesh
natsort
gradio
einops
```

Run `$ENV_BIN/pip install -r /tmp/requirements.txt`, then remove that temporary file. Keep the existing system libraries, Miniforge installation, CUDA environment variables, cleanup, `%environment`, `%help`, and `%runscript`.

- [ ] **Step 3: Verify the self-contained definition statically**

Run: `! rg -n '^%files|environment\.yml|mamba env create|flash-attention\.git' utonia.def && rg -n 'cuda:12\.6\.3-devel|CONDA_OVERRIDE_CUDA=12\.6|torch==2\.4\.1|torchvision==0\.19\.1|torchaudio==2\.4\.1|torch-2\.4\.0\+cu124|spconv-cu124|flash_attn-2\.6\.3\+cu126torch2\.4-cp310-cp310-linux_x86_64\.whl|h5py|camtools|--bind "\$PWD:/workspace"' utonia.def`

Expected: the first command succeeds with no output; the second lists CUDA 12.6.3, the Torch 2.4.1 stack, the exact FlashAttention binary wheel, every required runtime package, and the external-source bind-mount command.

- [ ] **Step 4: Check definition structure and whitespace**

Run: `awk '/^%/{print NR ":" $0}' utonia.def && git diff --check`

Expected: sections are `%post`, `%environment`, `%help`, and `%runscript`, with no whitespace errors.

- [ ] **Step 5: Commit**

```bash
git add utonia.def
git commit -m "build: inline Utonia container dependencies"
```

### Task 2: Validate on the remote GPU cluster

**Files:**
- Verify: `utonia.def`

**Interfaces:**
- Consumes: an Apptainer-enabled NVIDIA GPU cluster and a Utonia source checkout.
- Produces: a built `utonia.sif` and successful imports inside the image.

- [ ] **Step 1: Build the image from the checkout root**

Run: `apptainer build utonia.sif utonia.def`

Expected: the build completes without reading `environment.yml`.

- [ ] **Step 2: Validate CUDA and Utonia extension imports**

Run:

```bash
apptainer exec --nv --bind "$PWD:/workspace" utonia.sif bash -lc '
  cd /workspace &&
  export PYTHONPATH=./ &&
  python -c "import torch, spconv.pytorch, torch_scatter; print(torch.__version__, torch.cuda.is_available())"
'
```

Expected: PyTorch reports `2.4.1`, CUDA availability is `True`, and both extensions import.

- [ ] **Step 3: Validate the trajectory-PCA CLI is discoverable**

Run:

```bash
apptainer exec --nv --bind "$PWD:/workspace" utonia.sif bash -lc '
  cd /workspace &&
  export PYTHONPATH=./ &&
  python demo/9_pca_joint_trajectory.py --help
'
```

Expected: the help command succeeds without loading a model or opening a viewer.
