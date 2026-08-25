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

## Documentation
- [Installation Guide](docs/installation.md)
- [Public API Reference](docs/api.md)

## Known Limitations
The following features are **not yet supported or verified** in V1.0.0:
- ABI3 Forward Compatibility (currently bound directly to CPython 3.13 on Windows).
- LiveKit Cloud deployment verification (only local LiveKit server is verified).
- CPU fallback / non-NVDEC software decoding.
- 4K resolution stream verification.

## License
Velo is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
