# Velo

**WebRTC video → GPU-native decode → PyTorch CUDA tensor**

Velo is a developer-facing bridge designed to stream realtime WebRTC video directly into GPU-resident AI-ready PyTorch tensors with **zero CPU video decoding** and **zero CPU-to-GPU memory copies**.

## Why Velo?

In traditional Python media pipelines, processing realtime video for AI models follows this path:
1. **Network**: Receive RTP packets.
2. **CPU Decode**: Assemble and decode H.264 frames on the CPU.
3. **GIL Bottleneck**: Frame decoding blocks the Python interpreter.
4. **Memory Copy**: Upload decoded pixel buffers from CPU RAM to GPU VRAM.

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

## Supported Environments
- **Platform**: Windows x64
- **Python**: CPython 3.13 (win_amd64)
- **Hardware**: NVIDIA GPU with NVENC/NVDEC capabilities
- **Software**: NVIDIA Display Driver supporting CUDA 12+, CUDA-enabled PyTorch

## Installation

You must install Velo's runtime prerequisites manually to avoid overwriting CUDA-enabled PyTorch environments.
Please refer to the detailed [Installation Guide](docs/installation.md) for step-by-step instructions.

```bash
pip install velo
```

## Minimal Usage

Velo provides a clean Python interface that wraps all low-level networking and hardware code.

```python
import velo
import torch

torch.cuda.init()

# Connect to a WebRTC stream (e.g. from an SDP offer)
stream, sdp_answer = velo.connect(sdp_offer)

with stream:
    while True:
        try:
            # Blocks and releases Python GIL internally
            frame = stream.next()
            
            # Map directly to PyTorch CUDA tensor (zero-copy, cuda:0, uint8, HxWxC)
            tensor = frame.to_torch()
            
            print(f"Shape: {tensor.shape}, Device: {tensor.device}")
            
        except velo.StreamClosedError:
            break
```

## LiveKit Integration

Velo natively supports subscribing to [LiveKit](https://livekit.io/) tracks. You must provide a valid WebSocket URL and Access Token.

```python
stream = velo.connect_livekit("ws://localhost:7880", livekit_token)
with stream:
    # Consume frames natively...
    frame = stream.next()
```

## Verified Performance (V1.11 Benchmarks)

All performance figures below are measured on real NVIDIA hardware across 3 independent iterations ($\mu \pm \sigma$):

### 1. Real Hardware NVDEC Video Decoding (Class 4)
*Measured on NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.8, PyTorch 2.7.1+cu128)*

| Resolution | Format | Hardware Decoder | Throughput ($\mu \pm \sigma$) | Decode Interval ($p_{50}$) | Zero-Copy Tensor Overhead |
|---|---|---|---|---|---|
| **480p** (640x480) | H.264 Baseline/High | NVIDIA NVDEC | **3,486.5 ± 217.7 FPS** | 0.26 ms | **10.5 ± 0.24 $\mu\text{s}$** |
| **720p** (1280x720) | H.264 Baseline/High | NVIDIA NVDEC | **1,695.1 ± 10.5 FPS** | 0.58 ms | **10.2 ± 0.22 $\mu\text{s}$** |
| **1080p** (1920x1080) | H.264 Baseline/High | NVIDIA NVDEC | **627.9 ± 186.8 FPS** | 1.15 ms | **10.3 ± 0.18 $\mu\text{s}$** |
| **4K UHD** (3840x2160) | H.264 Baseline/High | NVIDIA NVDEC | **121.1 ± 27.4 FPS** | 7.27 ms | **9.9 ± 0.15 $\mu\text{s}$** |

> [!NOTE]
> Converting a GPU-resident `Frame` to a PyTorch CUDA tensor (`frame.to_torch()`) is a zero-copy DLPack memory pointer wrap taking **$\approx 10\, \mu\text{s}$** with zero CPU roundtrips.

### 2. Audio Processing & Synchronization (Class 2 & 4+2)
- **Audio Ingestion Throughput**: **76,868 ± 7,851 chunks/sec** (48 kHz stereo PCM)
- **Voice Activity Detection (VAD)**: $p_{50} = \mathbf{269.6\, \mu\text{s}}$
- **Temporal Context Lookup Latency**: $p_{50} = \mathbf{3.67\, \mu\text{s}}$
- **Audio/Video Timestamp Correlation Skew**: Mean **113.9 ms**, $p_{95} = \mathbf{368.3\, ms}$

For benchmark methodology and classification criteria, see [docs/v1.11-benchmark-methodology.md](docs/v1.11-benchmark-methodology.md).

## Documentation
- [Installation Guide](docs/installation.md)
- [Public API Reference](docs/api.md)
- [Benchmark Methodology & Taxonomy](docs/v1.11-benchmark-methodology.md)
- [Observability Architecture](docs/v1.10-observability-architecture.md)

## Known Limitations
- Real SmolVLM model inference requires $\ge 6.0\, \text{GB}$ VRAM for weights + KV cache in bfloat16; systems with $\le 4\, \text{GB}$ VRAM use `MockMultimodalAdapter` for pipeline infrastructure workloads.
- LiveKit Cloud deployment automated E2E tests skip gracefully when credentials (`LIVEKIT_URL`, `LIVEKIT_API_KEY`) are omitted from the local environment.

## License
Velo is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
