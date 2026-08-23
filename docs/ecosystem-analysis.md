# Velo: Ecosystem and Architecture Analysis

This document analyzes existing technologies in the realtime media and AI inference space to clarify Velo's intended differentiation and ensure we are not reinventing existing solutions.

## 1. Technology Analysis

### NVIDIA DeepStream
*   **What problem does it solve?** Enterprise-scale GPU-accelerated video analytics pipelines.
*   **Who uses it?** Smart cities, retail analytics, factory automation.
*   **Main abstractions:** GStreamer pipelines, plugins, metadata graphs.
*   **WebRTC support?** Not natively. Requires complex custom GStreamer `webrtcbin` integration or external Python servers.
*   **GPU decoding?** Yes (NVDEC).
*   **Frame memory location:** GPU (NVMM).
*   **Exposed to AI frameworks?** Yes, via Python bindings (`pyds`) and `BufferRetriever` exporting DLPack.
*   **Multi-stream?** Highly optimized for RTSP multi-stream.
*   **Developer configuration:** Building GStreamer pipelines is notoriously complex and un-Pythonic.
*   **Overlap with Velo:** GPU decoding, multi-stream handling, tensor handoff.
*   **Non-overlap:** DeepStream focuses on RTSP/analytics pipelines, not simple WebRTC-to-Tensor developer ergonomics.

### NVIDIA Holoscan
*   **What problem does it solve?** Real-time streaming AI pipelines for edge computing (medical, sensors).
*   **Main abstractions:** Operators, fragments, message passing, tensors.
*   **WebRTC support?** Supported via `webrtc_client` operator in HoloHub.
*   **GPU decoding?** Yes.
*   **Frame memory location:** GPU.
*   **Exposed to AI frameworks?** Yes, native DLPack support (`Holoscan Tensor`).
*   **Developer configuration:** Requires building a directed acyclic graph (DAG) of operators.
*   **Overlap with Velo:** Python APIs, DLPack integration, WebRTC ingestion.
*   **Non-overlap:** Holoscan is a general-purpose graph framework. Velo aims to be a simpler, focused media runtime.

### NVIDIA PyNvVideoCodec
*   **What problem does it solve?** Pythonic access to hardware video codecs (NVDEC/NVENC).
*   **Main abstractions:** `SimpleDecoder`, `SimpleEncoder`.
*   **WebRTC support?** No. It is strictly a codec library.
*   **GPU decoding?** Yes.
*   **Frame memory location:** GPU.
*   **Exposed to AI frameworks?** Yes, outputs DLPack which can be directly converted via `torch.from_dlpack()`.
*   **Developer configuration:** The developer must handle all networking, packetization, and synchronization.
*   **Overlap with Velo:** Hardware decoding and DLPack handoff. (Velo will likely *use* this internally).
*   **Non-overlap:** It does not handle transport (WebRTC).

### GStreamer & FFmpeg
*   **What problem do they solve?** Generic multimedia pipeline construction and transcoding.
*   **WebRTC support?** GStreamer supports `webrtcbin`; FFmpeg lacks native P2P signaling.
*   **GPU decoding?** Yes, both support hardware acceleration.
*   **Exposed to AI frameworks?** Very difficult in Python. Usually involves CPU copies (e.g., via PyAV) or writing custom C/C++ plugins.
*   **Overlap with Velo:** Media decoding.
*   **Non-overlap:** They are not AI-first. Extracting zero-copy GPU tensors in Python is extremely cumbersome.

### LiveKit
*   **What problem does it solve?** Scalable WebRTC SFU and real-time audio/video infrastructure.
*   **Main abstractions:** Rooms, Participants, Tracks, Agents.
*   **GPU decoding?** The LiveKit Agent framework pulls tracks, but frames are typically decoded on the CPU (using `aiortc` or similar) unless custom-configured.
*   **Exposed to AI frameworks?** Usually via CPU NumPy arrays, which are then copied to the GPU for PyTorch.
*   **Overlap with Velo:** WebRTC transport.
*   **Non-overlap:** LiveKit solves the *network* problem. Velo aims to solve the *node-level GPU processing* problem. (Velo could serve as the processing engine inside a LiveKit Agent).

### DLPack
*   **What problem does it solve?** Standardized in-memory tensor structure for zero-copy sharing between frameworks (e.g., PyTorch, CuPy).
*   **Overlap with Velo:** Velo will use DLPack as the standard handoff mechanism to AI inference.

---

## 2. Synthesis and Product Strategy

### Confirmed Gaps (Why Velo?)
The analysis confirms the core hypothesis: **There is a missing developer-facing abstraction for WebRTC-to-GPU-Tensor.**
*   If a developer wants WebRTC to PyTorch today, they usually use `aiortc` (CPU decode) $\rightarrow$ NumPy (CPU RAM) $\rightarrow$ PyTorch (GPU transfer). This is slow.
*   If they want zero-copy GPU decoding, they must use DeepStream (complex GStreamer), Holoscan (complex DAGs), or wire up PyNvVideoCodec to a custom WebRTC stack manually in C++.
*   **Velo's value is developer ergonomics:** Wrapping PyNvVideoCodec + WebRTC + DLPack into a clean `stream.next()` Python API.

### Features We Should NOT Build
*   **Do not build a new WebRTC SFU:** LiveKit already does this perfectly. Velo should ingest WebRTC tracks, not route them between users.
*   **Do not build a new hardware decoder:** Use `PyNvVideoCodec` or NVDEC directly.
*   **Do not build a new tensor format:** Use DLPack.
*   **Do not build a DeepStream clone:** Avoid exposing GStreamer pipelines to the user.

### Proposed Velo V1 Focus
Velo should act as a bridge. The ideal V1 stack:
1.  **Ingestion:** Ingest WebRTC track (RTP packets).
2.  **Decoding:** Pass encoded packets to `PyNvVideoCodec` (GPU NVDEC).
3.  **Handoff:** Expose decoded frames as DLPack.
4.  **Inference:** Consume DLPack in PyTorch.

### Biggest Technical Risks
1.  **WebRTC to NVDEC integration:** Extracting encoded H.264/VP8 payloads from WebRTC RTP packets in Python fast enough to feed `PyNvVideoCodec` without falling back to CPU decoding libraries like `av`.
2.  **Concurrency:** Managing asyncio (often used for WebRTC) alongside CUDA stream synchronization without deadlocks.

### Recommended Next Implementation Step
Before building the full runtime, we must write a **V0.1 experimental script** that attempts to parse a raw WebRTC video track, extract the encoded bitstream, and decode it using `PyNvVideoCodec` into a PyTorch tensor. This will validate the most critical technical risk.
