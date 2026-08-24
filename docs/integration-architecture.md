# Velo Integration & Scaling Architecture

This document studies the design constraints, memory models, and API interfaces for scaling Velo to multi-stream pipelines and third-party WebRTC backends (like LiveKit).

---

## 1. LiveKit Integration Design Study

### LiveKit Architecture Overview
LiveKit handles WebRTC session orchestration, signaling, room management, and media forwarding via their server. The LiveKit Agent SDK (written in Python) receives incoming user video tracks.

### Design Options for Velo Integration

#### Option A: Explicit Track Receiver (Mock/Synthetic Testing Only)
Used to verify track ingestion and queue structures using mock track/packet objects. The official LiveKit Python SDK FFI layer abstracts and decodes video tracks natively inside its C++ engine, exposing only decoded CPU frames to Python. Thus, this option cannot be used for E2E zero-copy LiveKit integration.

#### Option B: Direct Peer Connection Ingestion (Direct WebRTC Ingestion)
Velo behaves as a separate WebRTC peer and connects directly to the LiveKit SFU. The signaling handshake is managed in Python using raw WebSocket and `livekit-protocol` Protobuf stubs, while the actual SDP answer generation, ICE connectivity, DTLS, and realtime H.264 media data plane are managed natively in Rust via `webrtc-rs` and NVDEC (using the standard `velo.connect` FFI wrapper). This completely bypasses CPU decoding and satisfies the non-negotiable GPU-native pipeline constraints.


---

## 2. Multi-Stream Ingestion Design Study

When processing multiple video channels concurrently on a single GPU, the pipeline faces scheduling and resource constraints:

### Technical Constraints
1. **Decoder Contexts**: Each H.264 stream requires its own `NvDecoder` context to track SPS/PPS and slice references. Decoders cannot share internal reference states.
2. **GPU Memory Allocation**: Spawning decoders allocates dedicated hardware surfaces. To avoid VRAM fragmentation, decoders must be pre-allocated or pooled.
3. **Queue Policy & Backpressure**: Realtime streams must never block the native Toko worker. A slow Python inference loop on Stream A must not block the decoding of Stream B. Therefore, **independent bounded drop-oldest queues** are required per stream.

### Recommended API Interface

We recommend retaining individual `Stream` instances for developer simplicity:
```python
# Create streams independently
stream1, _ = velo.connect(sdp_offer1)
stream2, _ = velo.connect(sdp_offer2)

# Python consumer loop
while True:
    frame1 = stream1.next() # blocks
    frame2 = stream2.next() # blocks
```
Or utilizing an asynchronous gather loop:
```python
# Prototyped Async-Gathering API
import asyncio

async def process_stream(stream):
    with stream:
        while True:
            frame = await stream.next_async() # releases GIL
            tensor = frame.to_torch()
            # run model inference
```

---

## 3. Ingestion Categorization Matrix

| Input Type | Classification | Rationale |
|---|---|---|
| **Native WebRTC (SDP)** | **CORE** | Standard loopback/local developer testing. Minimal overhead. |
| **Custom RTP Push** | **CORE** | Crucial low-level entry point for integrations (LiveKit, RTSP, GStreamer). |
| **LiveKit Adapter** | **INTEGRATION** | Build as a Python wrapper in `velo.adapters.livekit` on top of RTP Push. |
| **RTSP Client** | **INTEGRATION** | Integrate via an external RTSP-to-WebRTC bridge, not natively inside Velo. |
| **WebRTC SFU Server** | **OUT OF SCOPE** | Velo is an endpoint consumer, not a media router. |
