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
**ARCHITECTURALLY PLAUSIBLE, RUNTIME UNVERIFIED** (V0.2-B Experiment)
* `webrtc-rs` is a high-performance port of Pion. It operates at a low level and exposes `TrackRemote.read_rtp()`, which yields raw network packets.
* **Architecture:** We successfully authored a Rust implementation utilizing `rtp::packetizer::Depacketizer` to parse Single NAL, STAP-A, and FU-A RTP payloads into H.264 Annex-B encoded bytes.
* **Limitation:** The runtime experiment failed strictly due to the host machine lacking the 4GB+ MSVC Visual Studio C++ Build Tools. Because compilation was blocked, we could not prove the integration with NVDEC at runtime.
* **Status:** This remains the recommended primary candidate, pending a proper build environment.

---

## 4. Current Architecture (Theoretical/Target)
Based on investigations, the target architecture is a hybrid native application:

1. **Network Layer (Rust):** `webrtc-rs` manages ICE/DTLS/SRTP and intercepts raw RTP video packets.
2. **Depacketization (Rust):** A Jitter Buffer (`webrtc::media::SampleBuilder`) reassembles H.264 Annex-B NAL units from RTP payloads.
3. **Python FFI (PyO3):** The Rust module is compiled as a native Python extension, exposing a zero-copy byte stream API (e.g., `velo.webrtc`).
4. **Hardware Decode (Python/C++):** `PyNvVideoCodec` consumes the NAL byte stream and triggers NVDEC.
5. **Inference (Python):** PyTorch consumes the resulting DLPack GPU tensors.

---

## 5. Current Unknowns & Future Decisions
* **WebRTC Correctness:** Does the custom Rust depacketizer robustly handle extreme packet loss, PLI/FIR generation, and dynamic resolution changes (keyframe requests) typical in WebRTC?
* **PyNvVideoCodec Stream Blocking:** Do operations on the resulting PyTorch tensors block safely on the NVDEC CUDA stream, or is manual CUDA synchronization required to prevent race conditions during continuous AI processing?
* **Build Distribution:** If `webrtc-rs` is the final choice, how do we distribute it without requiring Python users to install MSVC and Rust? (Pre-compiled wheels will be required).
