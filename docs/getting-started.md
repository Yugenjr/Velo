# Getting Started with Velo

Velo is a developer-facing bridge that allows you to stream WebRTC video directly into GPU-native PyTorch CUDA tensors without CPU video decoding or CPU-to-GPU memory round-trips.

## Prerequisites

To run Velo, your system must meet the following hardware and software requirements:

### Hardware Requirements
- **NVIDIA GPU**: Required for NVDEC (NVIDIA Hardware Video Decoder) and CUDA tensor execution. Turing architecture (RTX 20-series, GTX 1660) or newer is recommended.

### Software Requirements
- **Operating System**: Windows is currently fully supported and validated. Linux support is in the roadmap.
- **Python**: version `3.8` or newer.
- **NVIDIA CUDA Toolkit**: Installed and configured (should match your PyTorch CUDA runtime, e.g. CUDA 12.x).
- **PyTorch**: Installed with CUDA support (`torch.cuda.is_available()` must return `True`).
- **PyNvVideoCodec**: The NVIDIA Video Codec SDK Python binding must be installed in your environment.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/Google/Velo.git
cd Velo
pip install -e .
```

This will invoke `maturin` to compile the native Rust code (`velo_native`) and link it into your Python environment.

## Your First Example

The easiest way to get started is by running the basic webcam example. This server uses `aiohttp` to host a simple webpage, captures your webcam stream, sends the WebRTC SDP offer to Velo, and prints decoded tensor shapes in real time.

1. Navigate to the examples directory:
   ```bash
   cd examples
   ```

2. Run the example server:
   ```bash
   python -u basic_webcam.py
   ```

3. Open your browser and navigate to `http://localhost:8080`.
4. Click **Start Streaming to Velo** and grant webcam permissions.
5. Watch the console logs on your terminal:

```text
[Python] PyTorch CUDA context initialized.
============================================================
  Velo 0.1.0 Basic Webcam Example Server
  1. Open http://localhost:8080 in your browser
  2. Click 'Start Streaming to Velo'
  3. Check console logs for GPU tensor shape and device
============================================================
[Python] Received WebRTC offer. Connecting native Velo core...
[Python] SDP negotiation successful. Launching frame consumer...
[Python] Started blocking frame processing loop (GIL released).
[Velo Example] Frame 30 | Shape: (480, 640, 3) (640x480) | Residency: cuda:0 | FPS: 30.1 | Dropped: 0
[Velo Example] Frame 60 | Shape: (480, 640, 3) (640x480) | Residency: cuda:0 | FPS: 30.0 | Dropped: 0
```

6. Click **Stop Stream** in the browser to trigger a graceful shutdown. Velo will release all CUDA contexts and WebRTC connections cleanly.

## Understanding the Code

Below is the core loop from the example showing the public Velo API:

```python
import velo
import torch

# 1. Establish connection by passing the browser's SDP offer
# Returns a Stream object and the local SDP answer to return to the browser
stream, sdp_answer = velo.connect(sdp_offer)

# 2. Consume frames using a context manager for automatic cleanup
with stream:
    while True:
        # Blocks and releases the Python GIL during network waiting
        frame = stream.next()
        
        # Zero-copy mapping to PyTorch CUDA tensor via DLPack
        tensor = frame.to_torch()
        
        # Access frame properties
        print(f"Shape: {frame.shape} | Resolution: {frame.width}x{frame.height}")
```

## Troubleshooting

### `VeloConnectionError: set_remote_description failed...`
This occurs if the SDP offer sent by the browser is malformed or lacks ICE credentials. Ensure your frontend RTCPeerConnection adds H.264 video tracks before calling `createOffer()`.

### `ImportError: DLL load failed...`
On Windows, `PyNvVideoCodec` relies on native CUDA DLLs. Make sure you initialize PyTorch CUDA first (`import torch; torch.cuda.init()`) in your script. This loads the required driver DLLs into the process address space before `PyNvVideoCodec` is imported.
