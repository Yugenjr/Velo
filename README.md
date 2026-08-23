# Velo

**WebRTC video → GPU-native decode → PyTorch CUDA tensor**

Velo is a developer-facing bridge designed to stream realtime WebRTC video directly into GPU-resident AI-ready PyTorch tensors with **zero CPU video decoding** and **zero CPU-to-GPU memory copies**.

## Why Velo?

In traditional Python media pipelines (e.g. using `aiortc` and `PyAV` or `OpenCV`), processing realtime video for AI models follows this path:

1. **Network**: Receive RTP packets.
2. **CPU Decode**: Assemble and decode H.264 frames on the CPU into numpy arrays.
3. **GIL Bottleneck**: Frame decoding blocks the Python interpreter.
4. **Memory Copy**: Upload decoded pixel buffers from CPU RAM to GPU VRAM.

This pipeline creates severe CPU bottlenecks and latency, limiting throughput for multi-stream vision models.

**Velo solves this by keeping the entire media data plane GPU-native:**

```text
Browser Camera (H.264)
       ↓  (WebRTC network bytes)
Rust Native Core (webrtc-rs)
       ↓  (Annex-B H.264 NALs)
NVIDIA Hardware Decoder (NVDEC / PyNvVideoCodec)
       ↓  (GPU-resident surface)
DLPack Capsule (zero-copy memory sharing)
       ↓  (PyO3)
PyTorch CUDA Tensor (cuda:0)
```

By bypassing CPU decoding and CPU-to-GPU RAM copies, Velo achieves high FPS and minimal latency, leaving the CPU completely free for other operations.

---

## Core Principles

- **Zero CPU Video Decoding**: H.264 NAL units are extracted natively and sent directly to NVDEC.
- **Zero-Copy Memory Boundary**: GPU frames are wrapped into DLPack `PyCapsule` objects, allowing PyTorch to consume the memory instantly.
- **GIL-Free Concurrency**: Concurrency is managed in native Rust via a background Tokio runtime. The Python GIL is released during blocking waits.
- **Developer Ergonomics**: A clean, 5-line Python interface wraps all low-level networking and hardware code.

---

## What Velo Is Not

Velo is **not** a replacement for:
- **GStreamer / FFmpeg**: It does not support arbitrary demuxing, software filtering, or complex audio/video transcoding.
- **NVIDIA DeepStream**: It is not a complete analytics SDK. It is a simple python media-to-GPU primitive.
- **WebRTC SFUs / Media Servers**: Velo is a receiver/endpoint node, not a multi-party router.

---

## Quick Start

### Installation

Ensure you have an NVIDIA GPU, the CUDA Toolkit, PyTorch (with CUDA support), and `PyNvVideoCodec` installed. Then, install Velo:

```bash
pip install .
```

### Usage Example

```python
import velo
import torch

# Initialize CUDA context
torch.cuda.init()

# Establish WebRTC stream from an SDP offer
stream, sdp_answer = velo.connect(sdp_offer)

# Consume GPU-resident frames using context manager
with stream:
    while True:
        try:
            # Blocks and releases Python GIL internally
            frame = stream.next()
            
            # Map directly to PyTorch CUDA tensor (zero-copy, cuda:0)
            tensor = frame.to_torch()
            
            # Access metadata
            print(f"Shape: {frame.shape} ({frame.width}x{frame.height})")
            
        except velo.StreamClosedError:
            break
```

For a complete working webcam example with a browser frontend, check out [examples/basic_webcam.py](examples/basic_webcam.py).

## Documentation
- [Getting Started & Installation Guide](docs/getting-started.md)
- [Technical Architecture](docs/architecture.md)

## License
Velo is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
