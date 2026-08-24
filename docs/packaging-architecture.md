# Velo V1.0 Packaging & Distribution Architecture

This document outlines the packaging architecture, dependencies, build-time requirements, and target platform matrix required to distribute Velo as a prebuilt Python wheel without requiring compilers or Rust tools on the user's system.

---

## 1. Current Packaging Architecture

Velo's codebase consists of a native Rust WebRTC receiver and packet depacketizer wrapper, coupled with a Python core client.
* **Build System**: Uses `maturin` as the PEP 517 build backend.
* **Rust extension**: Compiled as a `cdylib` (`_velo_native.so` on Linux, `_velo_native.pyd` on Windows).
* **Signaling & Python API**: Implemented in Python under `src/velo` (imports the native extension and interacts with `PyNvVideoCodec` via Python-level object wrappers).

### Key Architectural Discovery
The native Rust code does **not** link directly to CUDA SDK libraries or C++ NVIDIA headers. Instead, it dynamically imports and interacts with `PyNvVideoCodec` at runtime using PyO3 bindings.
This design decouples Velo's Rust extension compilation from the CUDA compiler toolchain, allowing standard Rust compilers to build wheels without needing CUDA Toolkit or C++ compilers on the build host.

---

## 2. Dependency Strategy

### Python Runtime Dependencies
1. **PyTorch (`torch`)**: Required for backing decoded GPU surfaces with CUDA tensors. PyTorch packages already bundle all necessary CUDA runtime libraries (`libcudart`, `libcublas`, etc.) internally in their PyPI wheels.
2. **PyNvVideoCodec**: The official NVIDIA high-performance video encoding/decoding bindings. Can be installed directly from PyPI.
3. **websockets**: Required for signaling connections with the LiveKit room.
4. **livekit-api & livekit-protocol**: Required for token generation and signaling protocol serialization.

### CUDA & NVIDIA Driver Requirements
* **System Display Driver**: The host must have a compatible NVIDIA Display Driver installed (providing `libcuda.so.1` on Linux/WSL2 or `nvcuda.dll` on Windows).
* **Bundling**: **No CUDA runtime or driver libraries should be bundled in the Velo wheel.** PyTorch bundles its own runtime libraries, and `PyNvVideoCodec` loads the system display driver dynamically via standard OS dynamic linking.

---

## 3. Proposed Wheel Matrix

We will support the stable ABI (`abi3`) to compile a single portable wheel per OS/architecture that is compatible with all Python versions >= 3.8.

### Target Platforms

| Platform | Arch | Target Tag | Dependencies |
| :--- | :--- | :--- | :--- |
| **Windows** | x86_64 | `win_amd64` / `cp38-abi3` | System NVIDIA Driver, PyTorch (CUDA), PyNvVideoCodec |
| **Linux** | x86_64 | `manylinux2014_x86_64` / `cp38-abi3` | System NVIDIA Driver, PyTorch (CUDA), PyNvVideoCodec |
| **WSL2** | x86_64 | `manylinux2014_x86_64` / `cp38-abi3` | WSL2 NVIDIA Driver mounting, PyTorch (CUDA), PyNvVideoCodec |

---

## 4. Build and CI/CD Design

We propose using **GitHub Actions + `cibuildwheel`** to automate the building of release wheels:

### Workflow Structure
1. **Trigger**: Runs on tag creation or manual release dispatch.
2. **Job Matrix**:
   * Windows runner (`windows-latest`) to compile `win_amd64` wheels.
   * Linux runner (`ubuntu-latest`) running inside `manylinux2014` docker images to compile portable Linux wheels.
3. **Rust Compiler**: Set up via `actions-rust-lang/setup-rust-toolchain`.
4. **Build Tool**: `cibuildwheel` picks up the PEP 517 build backend (`maturin`) and handles cross-Python version packaging and stable ABI setup.

---

## 5. Clean-Install Test Strategy

A wheel will only be considered **VERIFIED** when imported and run in a clean, non-development sandbox environment.

### Verification Steps
1. Create a fresh virtual environment:
   ```bash
   python -m venv clean_env
   source clean_env/bin/activate
   ```
2. Install the built wheel locally:
   ```bash
   pip install velo-0.1.0-cp38-abi3-manylinux2014_x86_64.whl
   ```
3. Verify that pip installs all transitively listed dependencies (`PyNvVideoCodec`, `torch`, `websockets`, `livekit-api`).
4. Execute the verification script:
   ```bash
   python -c "import velo; print(velo.__file__)"
   ```
5. Run unit tests (`tests/test_unit.py`) to confirm PyO3 bindings work cleanly.

---

## 6. Licensing and Distribution Considerations

* **Velo License**: Distributed under the MIT License (already in the root).
* **NVIDIA PyNvVideoCodec**: MIT Licensed, freely distributable on PyPI.
* **Rust Dependencies**: `webrtc-rs`, `tokio`, `crossbeam`, and `bytes` are all distributed under permissive licenses (MIT or Apache 2.0).
* **Conclusion**: No licensing or distribution conflicts prevent publishing Velo on PyPI.

---

## 7. Risks and Mitigation

1. **Host-Target Compatibility (GLIBC)**:
   * *Risk*: Building Linux wheels on a modern Ubuntu host results in dependency on a new GLIBC version, causing crashes on older systems.
   * *Mitigation*: Enforce `manylinux2014` build containers in `cibuildwheel` to target GLIBC 2.17.
2. **Stable ABI (`abi3`) Compilation**:
   * *Risk*: Native Rust symbols might not align with older Python runtimes if features are not configured correctly.
   * *Mitigation*: Add `abi3-py38` features to PyO3 in `Cargo.toml` and configure maturin accordingly.
3. **NVIDIA Driver Mismatch**:
   * *Risk*: Users with outdated GPU drivers might crash during `PyNvVideoCodec` initialization.
   * *Mitigation*: Retain strict hardware audits inside Velo's `__init__.py` to warn users cleanly about driver version mismatch.
