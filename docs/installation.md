# Velo Installation Guide

This document describes the correct installation and configuration strategy for Velo V1.0.

---

## 1. Supported Environments

* **OS**: Windows x64 (10/11) / Linux x86_64
* **GPU**: NVIDIA NVENC/NVDEC capable GPU
* **Driver**: NVIDIA Display Driver supporting CUDA 12+
* **Python**: CPython 3.8+ (64-bit)

---

## 2. Installation Strategy & Prerequisites

Velo does **not** declare PyTorch or `PyNvVideoCodec` in its default pip package requirements to avoid corrupting or overwriting existing CUDA environments with CPU-only packages.

You must install Velo's runtime prerequisites manually before installing the Velo package.

### Step 1: Install CUDA-Enabled PyTorch
Verify that you are installing a version of PyTorch with active CUDA compilation. Standard PyPI downloads install CPU-only versions by default.

* **Command**:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu124
  ```
* **Verify**:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
  *(Must print `True`. If `False`, Velo will raise a `HardwareError` during initialization.)*

### Step 2: Install PyNvVideoCodec
Install the official NVIDIA Python bindings for NVDEC/NVENC.
* **Command**:
  ```bash
  pip install PyNvVideoCodec
  ```
* **Verify**:
  ```bash
  python -c "import torch; import PyNvVideoCodec; print(PyNvVideoCodec.__version__)"
  ```
  *(Note: You must import `torch` before `PyNvVideoCodec` to allow correct `cudart64_12.dll` library path mapping.)*

### Step 3: Install Velo
Now install the pre-compiled Velo wheel or package.
* **Command**:
  ```bash
  pip install velo
  ```

---

## 3. Real LiveKit E2E Verification
To execute real WebRTC streaming E2E checks, you must configure your LiveKit SFU parameters in the environment:
* `LIVEKIT_URL`
* `LIVEKIT_API_KEY`
* `LIVEKIT_API_SECRET`
