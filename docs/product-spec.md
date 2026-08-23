# Velo Product Specification (V0.7)

This specification defines the product identity, target market, competitive positioning, and technical roadmap for Velo following the runtime verification of the V0.5/V0.6 native WebRTC-to-GPU pipeline.

---

## 1. Product Definition
Velo is a developer-focused, high-performance native runtime that bridges live WebRTC video feeds directly to GPU-resident AI processing. It abstracts away RTP depacketization, jitter buffering, and NVDEC hardware decoding, yielding a zero-copy PyTorch CUDA tensor on `cuda:0` via DLPack.

---

## 2. Target Users

| Rank | Persona | Primary Goal | Existing Pain Point | Velo Value |
|---|---|---|---|---|
| **1** | **Interactive AI/VLM Developers** | Build realtime, low-latency vision agents (e.g. conversational VLMs, visual assistants). | High latency caused by CPU decoding and GIL locks; complex WebRTC-to-GPU integration boilerplate. | Delivers zero-copy GPU tensors directly from browser webcams in a 5-line Python API. |
| **2** | **Computer Vision Prototypers** | Quickly test new models (YOLO, segmentation) on live webcam or RTSP WebRTC streams. | Writing boilerplate WebRTC signaling, H.264 depacketization, and PyTorch copy code. | Eliminates network and decoding boilerplate; standardizes frame extraction. |
| **3** | **Robotics/Edge AI Developers** | Stream robot camera feeds back to server/edge GPUs for control loop inference. | Running high-overhead CPU decode pipelines on resource-constrained edge machines. | Bypasses CPU decoding entirely, saving CPU cycles for control systems or downstream AI. |
| **4** | **Video Analytics Developers** | Process multiple live surveillance streams on edge or cloud servers. | Building complex C++ pipelines or struggling with GStreamer configurations. | Simpler pythonic scripting boundary, though not optimized for high-density streams. |

---

## 3. Ranked Use Cases

### Rank 1: Browser Webcam → Realtime VLM / Interactive Vision Agents
- **Who needs it**: Developers building conversational or real-time reactive AI agents (e.g., visual coaching, automated drive-thrus, smart home assistants).
- **Existing Architecture**: WebRTC client → Node/Python signaling → WebRTC library (aiortc) → CPU H.264 Decode (PyAV/FFmpeg) → Host memory image → CUDA Copy → PyTorch VLM.
- **Pain Point**: Heavy CPU bottleneck due to GIL contention and CPU-GPU transfer latency, rendering real-time interaction laggy and resource-intensive.
- **Velo's Fit**: Replaces the CPU decode and CPU-to-GPU copy steps. The feed lands directly on `cuda:0` from the network.
- **Technical Feasibility**: High. Already demonstrated in the V0.5 integration test.

### Rank 2: LiveKit → Velo → GPU Inference
- **Who needs it**: Developers using LiveKit for multi-agent voice/video coordination who need to feed incoming user video tracks into GPU models.
- **Existing Architecture**: LiveKit Agent SDK → `aiortc` track receiver → CPU decodes → GPU transfer.
- **Pain Point**: LiveKit's agent framework is written in Python; running heavy AI models alongside CPU decoding limits the number of concurrent sessions per server.
- **Velo's Fit**: Connects natively to LiveKit WebRTC packet streams and decodes directly to GPU memory, scaling agent density.
- **Technical Feasibility**: Moderate. Requires exposing an RTP/Track ingestion API in Velo.

### Rank 3: Robotics Camera → WebRTC → Edge GPU Inference
- **Who needs it**: Robotics engineers teleoperating machines or executing centralized GPU-in-the-loop navigation.
- **Existing Architecture**: WebRTC stream over LAN → ROS node → CPU decode → CUDA copy → PyTorch.
- **Pain Point**: High CPU usage on the robot or local edge controller, leading to thermal throttling and dropped frames.
- **Velo's Fit**: Provides native zero-copy decoding on edge NVIDIA hardware (Jetson, edge PC).
- **Technical Feasibility**: High.

---

## 4. Competitive Analysis

### DeepStream Comparison
*Why choose Velo over DeepStream?*
- **DeepStream is better at**: High-density multi-channel datacenter streams, object tracking, TensorRT optimized pipelines, and RTSP ingestion.
- **DeepStream's pain point**: Extremely steep learning curve. Developers must write GStreamer pipelines or deal with complex C++ wrappers and restrictive Python bindings. It does not integrate ergonomically with standard PyTorch VLM models or custom python libraries.
- **Velo's niche**: Prototyping and building interactive single-to-few stream AI agents where the developer wants to write pure Python/PyTorch code without knowing GStreamer.
- **Verdict**: Do not compete with DeepStream for large-scale surveillance pipelines. Target interactive, low-latency, python-first AI development.

### PyNvVideoCodec Comparison
*Why choose Velo over PyNvVideoCodec directly?*
- **PyNvVideoCodec solves**: Raw bitstream file decoding into GPU-resident frames.
- **PyNvVideoCodec lacks**: Network streaming integration, WebRTC negotiation, ICE gathering, and RTP depacketization.
- **Velo's addition**: Velo handles the WebRTC session, depacketizes RTP streams into Annex-B H.264, and dynamically feeds them into PyNvVideoCodec's decoder.
- **Verdict**: Velo is not competing with PyNvVideoCodec; Velo wraps it to bridge the network-to-GPU boundary.

### LiveKit Comparison
- **LiveKit solves**: Multi-user rooms, SFU transport, client SDKs, and state synchronization.
- **Velo's addition**: Velo serves as the media plane worker that decodes incoming video tracks directly into GPU memory for AI processing.
- **Verdict**: Velo should not build SFUs or media servers. Velo should become the optimal GPU media processor integration for LiveKit agents.

### aiortc Comparison
- **aiortc solves**: Python-native WebRTC connectivity.
- **aiortc lacks**: Zero-copy GPU pipelines. It forces CPU video decoding and CPU-to-GPU copies.
- **Velo's addition**: Velo moves the entire media plane out of the Python interpreter, executing decoding in native Rust/Tokio and dropping frames straight onto the GPU.

---

## 5. Architectural Invariants

The Velo core must preserve the following guidelines:
1. **Native Media Plane**: WebRTC and RTP depacketization must execute entirely in native memory (Rust/Tokio background thread).
2. **Zero CPU Decode**: H.264 bytes must never be decoded into host CPU memory in the media path.
3. **Zero Host-to-Device Copy**: Frames must enter GPU memory during hardware decoding and stay there.
4. **DLPack Interoperability**: GPU memory must be transferred to Python via zero-copy DLPack capsules.
5. **No DeepStream Clone**: Do not implement GStreamer pipelines, multiplexers, or custom server dashboards.
6. **No Model Serving inside Velo**: Velo is a media-to-GPU pipe; model inference is left to PyTorch/VLM frameworks.

---

## 6. Functional Roadmap

### Now: Velo 0.1.0 (Current Baseline)
- Single-stream WebRTC input.
- H.264 Annex-B depacketization.
- PyNvVideoCodec NVDEC decoding.
- Bounded drop-oldest frame queue.
- Idempotent stream close and worker thread joining.
- Python `width`, `height`, `shape`, and `timestamp` properties on `Frame`.

### Next: Velo 0.2.0 (Developer Experience & Platform Expansion)
- **Linux Support**: Ensure native Rust bindings and Maturin builds work out-of-the-box on Ubuntu (vital for Jetson and server deployment).
- **RTSP WebRTC Bridge / Custom Track API**: Allow developers to feed RTP packets from arbitrary WebRTC sources (like LiveKit or RTSP-to-WebRTC streams) directly into the decoder.
- **Observability**: Expose detailed telemetry APIs (e.g. latency metrics, queue drop rates).

### Later: Velo 0.3.0 (Codec Expansion & Multi-Stream)
- **VP8/VP9 Codec Support**: Add VP8/VP9 decoding (crucial for native browser WebRTC clients).
- **Multi-Stream Support**: Allow running multiple NativeStream objects safely on the same CUDA context without context switching overhead.

---

## 7. Categorization Matrix

```mermaid
graph TD
    subgraph Core (Category A)
        WebRTC["WebRTC & RTP Handling"]
        Depack["H264/VP9 Depacketization"]
        NVDEC["NVDEC GPU Decoding"]
        DLPack["DLPack zero-copy"]
        Queue["Realtime Bounded Queue"]
    end
    subgraph Integrations (Category B)
        LiveKit["LiveKit Agent SDK"]
        PyTorch["PyTorch Helper Utils"]
        RTSP["RTSP-to-WebRTC source"]
    end
    subgraph Out of Scope (Category C)
        SFU["WebRTC SFU Server"]
        Serving["VLM / Model Serving"]
        Muxing["MP4/FLV Video Recording"]
        Dashboard["Monitoring Dashboards"]
    end
```

---

## 8. Benchmark Strategy

To prove Velo's technical differentiation, we will establish a benchmark suite measuring:

1. **End-to-End Latency**: The time from frame capture at the browser webcam to the tensor being available on `cuda:0`.
2. **CPU Utilization**: CPU overhead comparison during active streaming.
3. **GIL Contention**: How much the frame processing loop blocks other Python threads.
4. **GPU VRAM Footprint**: VRAM overhead of the decoder context.

### Target Comparisons:
- **Pipeline A (Velo)**: WebRTC H264 → Rust depack → NVDEC → DLPack → PyTorch CUDA.
- **Pipeline B (CPU Route)**: WebRTC H264 → aiortc (PyAV CPU decode) → Host memory → Tensor CUDA upload → PyTorch CUDA.

---

## 9. Open-Source Adoption Strategy

### The "5-Minute Test"
A developer should be able to:
1. `pip install velo`
2. Run `python examples/basic_webcam.py`
3. Connect their browser webcam and see PyTorch CUDA tensors print shape and GPU device information within 5 minutes.

### Key Adoption Blockers & Fixes:
1. **CUDA Installation Friction**: Developers often have mismatching CUDA versions. **Fix**: Provide clear dependency checks and explicit error messages (e.g., catching DLL load failures).
2. **Missing Pre-built Wheels**: Compiling Rust from source requires Visual Studio build tools (Windows) or compiler toolchains. **Fix**: Publish pre-compiled binary wheels for common Python/CUDA configurations on Windows/Linux.
3. **Signaling Complexity**: Standard WebRTC signaling is painful. **Fix**: Provide a lightweight HTTP/WebSocket signaling utility class inside `velo.utils` for quick prototyping.
