# Velo Linux Support Status

This document defines the compile-time and runtime requirements for running Velo on Linux environments (specifically Ubuntu + NVIDIA GPU).

---

## Verified vs. Unverified Status

> [!WARNING]
> **UNVERIFIED**: The Velo team has **not** executed runtime tests of the H.264 WebRTC data plane on Linux due to local hardware testing constraints (current tests are executed on a Windows 11 host with RTX hardware).
> 
> All compilation guidelines, dependency requirements, and system libraries listed below are **expected/architecturally plausible** based on Maturin/Rust/PyTorch conventions, but remain **unverified at runtime on Linux**.

---

## 1. Compilation Requirements (Build-Time)

To compile Velo from source on Linux, the following toolchains are required:

- **Rust Toolchain**: `rustup` must be installed. The default stable compiler (1.75+) is expected to compile `webrtc-rs` and `pyo3` bindings.
- **Maturin**: Used as the PEP 517 build backend. Installable via `pip install maturin` or `cargo install maturin`.
- **Compiler Toolchain**: GCC and Clang compilers must be present (`sudo apt install build-essential`).
- **System Cryptography Headers**: `webrtc-rs` depends on OpenSSL wrapper libraries. You must install development headers:
  ```bash
  sudo apt install pkg-config libssl-dev
  ```

---

## 2. Runtime Requirements

Running Velo requires a compatible hardware acceleration environment:

- **NVIDIA GPU**: Required for hardware NVDEC (NVIDIA Hardware Video Decoder) support.
- **NVIDIA Driver**: Proprietary drivers must be installed (recommended version `525` or newer). Verified via:
  ```bash
  nvidia-smi
  ```
- **CUDA Toolkit**: Install a version matching your target PyTorch build (e.g. CUDA 12.1 or 12.4).
- **PyTorch**: Install a CUDA-enabled version of PyTorch:
  ```python
  import torch
  assert torch.cuda.is_available() == True
  ```
- **PyNvVideoCodec**: The official NVIDIA Video Codec SDK Python binding must be installed in the environment.

---

## 3. Installation Flow (Expected)

Once all build-time dependencies are met, Velo should install cleanly using pip:

```bash
git clone https://github.com/Google/Velo.git
cd Velo
pip install -e .
```

This will automatically trigger the Maturin build system, compile the native `_velo_native` library, and register it as an editable python package.
