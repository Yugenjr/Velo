# Velo Architecture

## 1. Product Goal
Velo's core product goal is to provide a zero-copy, GPU-native WebRTC multimodal AI ingestion pipeline. The strict non-negotiable architectural requirement is:

```text
Realtime WebRTC media
        ↓
encoded media (H.264 NAL units)
        ↓
hardware GPU decode (NVDEC)
        ↓
GPU-resident frame
        ↓
GPU tensor / AI multimodal inference (PyTorch)
```

The system must NEVER perform CPU-based video decoding. CPU memory copies of encoded payloads (network buffers) are acceptable, but decoded pixels must remain exclusively on the GPU.

---

## 2. Validated GPU Backend
**VERIFIED** (V0.1-A Experiment: `nvdec_pytorch.py`)

We have successfully proven that `PyNvVideoCodec` can take encoded H.264 media and decode it via NVDEC directly into GPU device memory. Using DLPack, these decoded frames can be converted into PyTorch `cuda:0` tensors with zero CPU copies.

---

## 3. Investigated WebRTC Backends

### aiortc (Python)
**REJECTED** (V0.1-B Experiment)
* `aiortc` delegates all media handling to `PyAV` (FFmpeg) internally.
* It forces CPU decoding before yielding frames to the Python application.
* **Finding:** It is structurally impossible to intercept raw encoded H.264 NAL units using standard `aiortc` APIs without rewriting its core C-extensions.

### GStreamer (Python/C)
**REJECTED FOR CURRENT ENVIRONMENT** (V0.2-A Investigation)
* We investigated `webrtcbin` + `nvv4l2decoder` for a pure GPU pipeline.
* **Finding:** On Windows with standard NVIDIA drivers, the Linux-centric V4L2/NVMM memory pipeline does not exist. The Windows GStreamer plugins rely on `d3d11h264dec`, which requires extremely complex D3D11 to CUDA interoperability layers to get the memory into PyTorch.
* While technically possible, building custom GStreamer C-plugins to bridge D3D11 textures into CUDA tensors for a simple feasibility check was rejected.

### webrtc-rs (Rust Native WebRTC)
1. **Native WebRTC → Encoded H264**
   - **Status:** **VERIFIED**
   - **Implementation:** Rust (`webrtc-rs`)
   - **Notes:** Capable of negotiating H.264 and reconstructing raw NAL units (SPS, PPS, IDR, Non-IDR) natively via RTP, without CPU video decoding. Build environment resolved natively on Windows with MSVC.

2. **Direct Encoded H264 → NVDEC**
   - **Status:** **VERIFIED**
   - **Implementation:** `PyNvVideoCodec.CreateDecoder` (bypassing `SimpleDecoder` and Demuxer)
   - **Notes:** Python NVDEC binding directly accepts raw encoded H.264 bitstream (`PacketData`) via memory pointers. Bypasses file containers and demuxers completely, decoding directly into DLPack-compatible GPU-resident surfaces (`cuda:0`).

---

## 4. Current Architecture (Theoretical/Target)
Based on investigations, the target architecture is a hybrid native application:

1. **Network Layer (Rust):** `webrtc-rs` manages ICE/DTLS/SRTP and intercepts raw RTP video packets.
2. **Depacketization (Rust):** A Jitter Buffer (`webrtc::media::SampleBuilder`) reassembles H.264 Annex-B NAL units from RTP payloads.
3. **Python FFI (PyO3):** The Rust module is compiled as a native Python extension, exposing a zero-copy byte stream API (e.g., `velo.webrtc`).
4. **Hardware Decode (Python/C++):** `PyNvVideoCodec` consumes the NAL byte stream and triggers NVDEC.
5. **Inference (Python):** PyTorch consumes the resulting DLPack GPU tensors.

---

## 5. Native Core → Python Boundary (PyO3)

**Status: VERIFIED & HARDENED (V0.5 Production Foundation)**

The abstraction boundary between Python and the native runtime is verified and hardened:
- **PyO3 Core**: A native Python extension (`velo_native`) bridges the two environments.
- **Async Runtime**: The Rust native core manages its own background `tokio` runtime running inside a dedicated OS worker thread. The Python GIL is released during `next_frame()` blocking waits.
- **Media Path**: WebRTC network bytes and H.264 packets remain entirely in native Rust memory. Encoded NAL units are batched by RTP timestamp and fed into NVDEC using raw C pointers.
- **GIL Optimization**: Rather than acquiring the GIL for every individual NAL packet, NALs are accumulated and decoded in batches per frame boundary, reducing GIL acquisition frequency to match frame rates (~30Hz).
- **GPU Frame Ownership**: Decoded frames cross the Python boundary as DLPack `PyCapsule` objects. The `velo.Frame` object maintains a reference to the native NVDEC decoder context, guaranteeing the CUDA context outlives the PyTorch tensor.
- **Graceful Shutdown**: Calling `stream.close()` triggers a thread-safe shutdown flag, closes the WebRTC peer connection, terminates the Tokio runtime, joins the worker thread (with Python GIL released), and drains any remaining queue frames under the GIL.
- **Error Model**: Codec registration, WebRTC handshakes, and NVDEC failures are propagated as clean Python exceptions (`VeloConnectionError`, `DecodeError`, `StreamClosedError`) without panicking the Rust process.
- **Frame Queue**: Bounded queue (`size=3`) drops stale frames (`Drop-Oldest`) under the GIL to prevent memory leaks and keep pipeline latency strictly real-time.

---

## 6. Current Unknowns & Future Decisions
* **WebRTC Correctness:** Does the custom Rust depacketizer robustly handle extreme packet loss, PLI/FIR generation, and dynamic resolution changes (keyframe requests) typical in WebRTC?
* **PyNvVideoCodec Stream Blocking:** Do operations on the resulting PyTorch tensors block safely on the NVDEC CUDA stream, or is manual CUDA synchronization required to prevent race conditions during continuous AI processing?
* **Build Distribution:** If `webrtc-rs` is the final choice, how do we distribute it without requiring Python users to install MSVC and Rust? (Pre-compiled wheels will be required).

---

## 7. V0.9 Verified Status Matrix

We categorize Velo's capabilities into the following strict status definitions:

### VERIFIED (Actually executed successfully with runtime evidence)
- **H.264 WebRTC Native Data Plane**: Browser WebRTC H.264 video -> native Rust `webrtc-rs` -> raw Annex-B NAL batching -> NVDEC GPU decode -> DLPack -> PyTorch CUDA tensor.
- **Direct RTP Ingestion (`velo.RtpReceiver`)**: Pushing raw H.264 RTP packet payloads directly to a native parser OS worker thread, decoding on GPU, and producing `cuda:0` PyTorch tensors (verified in [tests/test_rtp.py](file:///c:/Users/Yugendra/Velo/Velo/tests/test_rtp.py)).
- **LiveKit Video Track Integration (Mock)**: Automated Async packet ingestion from a mock LiveKit video track, piped to native `RtpReceiver` (verified in [tests/test_livekit.py](file:///c:/Users/Yugendra/Velo/Velo/tests/test_livekit.py)).
- **Multi-Stream Scalability (4 Streams)**: Spawning 4 parallel thread consumers decoding independent streams concurrently, verifying frame count and queue drops under a 4-stream load (verified in [tests/test_multistream_4.py](file:///c:/Users/Yugendra/Velo/Velo/tests/test_multistream_4.py)).
- **Max Performance Baseline**: Deterministic high-throughput test decoding H.264 packets at **4,750+ FPS** hardware maximum throughput ceiling, with only **3.5% CPU** load under a targeted 30 FPS workload (verified in [tests/benchmark_rtp.py](file:///c:/Users/Yugendra/Velo/Velo/tests/benchmark_rtp.py)).
- **Unacceptable Codec Rejection**: Explicit rejection of unsupported/invalid codecs with clean ValueError propagation (verified in [tests/test_unit.py](file:///c:/Users/Yugendra/Velo/Velo/tests/test_unit.py)).
- **Linux Native Validation via WSL2**: Compiled and executed natively in a WSL2 Ubuntu 22.04 LTS VM with RTX 3050 Laptop GPU passthrough, resolving package bindings and PyTorch linker SIGBUS crash issues.
- **Precompiled Wheel Installation**: Maturin-compiled PEP-517 compliant binary wheels built and successfully verified in clean, compiler-free virtual environments on both Windows and Linux.

### UNVERIFIED / BLOCKED
- **Real LiveKit Server Direct Connection**: Direct WebRTC room track subscription and ingestion is implemented and experimentally verified on the signaling layer (via Websockets and Protobuf protocol stubs), but is unverified E2E due to missing `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` credentials in the test execution environment.


### PLANNED (Intentionally not yet implemented)
- **VP8, VP9, AV1, and HEVC/H.265 Codec Ingest**: Codec structures are modeled in the pipeline interfaces but remains planned for implementation.

