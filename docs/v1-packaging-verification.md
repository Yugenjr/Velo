# Velo V1.0 Packaging & Clean Install Verification Report (Windows Host)

This report details the final outcomes of Velo's V1.0 Packaging and installation verification.

---

## Final Verification Classifications

### VERIFIED
* **Windows Wheel Build**: Successfully built wheel tag: `velo-0.1.0-cp313-cp313-win_amd64.whl` using Maturin.
* **Native Extension Loading**:
  ```powershell
  python -c "import velo; import velo._velo_native; print(velo._velo_native)"
  ```
  *Result*: Successfully loaded extension binary `_velo_native.cp313-win_amd64.pyd` on CPython 3.13.
* **Dependency Metadata**: PyTorch and `PyNvVideoCodec` were deliberately excluded from standard package dependencies in `pyproject.toml` to prevent package managers from overwriting CUDA PyTorch environments with CPU-only wheels.
* **Unit Tests**: Executed `python tests/test_unit.py` and confirmed all 10 tests passed (including the new `test_custom_dimensions` test).
* **Host GPU Smoke Test**: Executed `python tests/test_livekit.py` and confirmed successful mock WebRTC track ingestion, NVDEC decoding, and mapping to `cuda:0` PyTorch tensors:
  ```text
  First LiveKit frame decoded successfully!
    Shape: (480, 640, 3)
    Residency: cuda:0
  [PASS] test_livekit_integration
  ```
* **4K Configuration Support**: Added optional `max_width` and `max_height` arguments to python connectors and constructors, enabling 4K (`3840`x`2160`) decoders configurations.
* **Real Local LiveKit E2E**: Executed `python tests/test_livekit_real.py` against the local LiveKit server running inside WSL2. Verified signaling, WebRTC negotiation, H.264 RTP reception, and NVDEC CUDA tensor outputs on the Windows host.

### FAILED
* **Standard Sandbox Installation (Without Pre-requisites)**: Clean sandbox `pip install velo` without pre-installing CUDA-enabled PyTorch correctly triggers import-time hardware sanity exceptions:
  ```text
  velo.exceptions.HardwareError: Velo requires a CUDA-capable NVIDIA GPU. torch.cuda.is_available() is False.
  ```

### BLOCKED
* **Real LiveKit Cloud E2E**: Blocked due to lack of real `LIVEKIT_URL` / `LIVEKIT_API_KEY` credentials on host environment.
* **Lifecycle Multi-Threaded Connection Test**: Connecting mock SDP handshakes on Windows triggers interface gathering blocks, halting test execution.

### UNIMPLEMENTED
* **ABI3 Stable API Support**: Maturin `abi3-py38` stable targeting is not yet implemented.
* **Real 4K Decode Verification**: No real 4K stream H.264 file was decoded during testing (only configuration support is verified).
