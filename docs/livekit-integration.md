# LiveKit E2E Integration & Signaling Architecture

This document specifies the LiveKit E2E integration design, resolving the official Python SDK's decoded-media limitation using a GPU-native direct WebRTC signaling handshake.

---

## 1. LiveKit Media Architecture & Handshake
LiveKit is a Selective Forwarding Unit (SFU). When a client joins a room, LiveKit establishes two separate WebRTC PeerConnections:
1. **Publisher PeerConnection**: Initiated by the client (client sends SDP Offer, server replies with SDP Answer) to publish local tracks.
2. **Subscriber PeerConnection**: Initiated by the server (server sends SDP Offer containing all active remote video/audio tracks in the room, client replies with SDP Answer).

Because Velo acts as a media consumer (ingester), it only needs to participate in the **Subscriber PeerConnection**.

---

## 2. Why the LiveKit Python SDK is Insufficient
The official `livekit` Python SDK handles all network packetization, jitter buffering, and media decoding internally inside its compiled Rust FFI core (`webrtc-sys` and native C++ `libwebrtc`).
* **Decoded Plane Only**: The public SDK APIs (like `rtc.VideoStream`) only yield raw YUV/RGBA CPU video frames.
* **No Encoded Access**: There are no public hooks or methods (such as `read_rtp()`) to obtain raw RTP packets or H.264 compressed bitstreams directly.
* **Architectural Conflict**: Feeding standard LiveKit tracks into Velo would require CPU-based re-encoding or a Host-to-GPU memory copy, violating Velo's non-negotiable **zero-CPU-decoding GPU-native pipeline**.

---

## 3. The Native Integration Boundary
To bypass C++ libwebrtc's internal CPU decoding, Velo connects directly to LiveKit's signaling WebSocket as a normal WebRTC peer, orchestrating the SDP exchange in Python while delegating the entire media transport and GPU decoding plane to Velo's native Rust core (`webrtc-rs` + NVDEC).

```text
       LiveKit SFU (Server)
                 ↓  (Signaling WebSocket)
  Protobuf SignalResponse (SDP Offer)
                 ↓
      Python Ingestion Orchestration
                 ↓
     sdp_offer string argument
                 ↓
        velo.connect() [Rust]
                 ↓
   Native PeerConnection (webrtc-rs)
                 ↓  (Gather complete)
        local sdp_answer string
                 ↓
      Python Ingestion Orchestration
                 ↓  (Signaling WebSocket)
  Protobuf SignalRequest (SDP Answer)
                 ↓
   LiveKit SFU establishes DTLS/ICE
                 ↓  (Realtime Media UDP)
    Raw H.264 RTP Packet Stream
                 ↓
    webrtc-rs Jitter Buffer (Rust)
                 ↓
    Annex-B NAL reassembly (Rust)
                 ↓
       NVIDIA NVDEC Decode (GPU)
                 ↓
       DLPack PyTorch CUDA Tensor
```

---

## 4. Exact Encoded-Media Path
1. **RTP Delivery**: LiveKit server streams H.264 encoded RTP packets directly to Velo's native media port over UDP.
2. **Jitter Buffer**: Velo's native `webrtc-rs` stack handles packet reordering, loss detection, and reassembly.
3. **Depacketization**: The Rust parser constructs H.264 Annex-B NAL units batched by frame boundary.
4. **Hardware Decoding**: batched NALs are passed via C-pointers directly to `PyNvVideoCodec` on the GPU.
5. **Zero-Copy Tensor**: PyTorch maps the decoded memory directly via DLPack into a `cuda:0` tensor.

---

## 5. Native Components Involved
- **`webrtc-rs` (Rust)**: Manages ICE candidate verification, DTLS handshakes, SRTP decryption, and RTP track reader loop (`track.read_rtp()`).
- **`rtp::codecs::h264` (Rust)**: Depacketizes H.264 NAL fragments natively.
- **`PyNvVideoCodec` (C++/Python)**: Calls NVIDIA NVDEC APIs to decode raw NAL units directly into GPU-resident frames.

---

## 6. Velo Integration Design
The integration is exposed via the unified `velo.connect_livekit(url, token)` function:
- **WebSocket Signaling Thread**: Python spawns a background asyncio event loop thread that connects to `ws://{host}/rtc` and exchanges Protobuf signaling messages (`SignalRequest`/`SignalResponse`).
- **Orchestration**: When the server sends a Subscriber SDP offer, Python passes it to `velo.connect(sdp_offer)` which initializes the native Rust PeerConnection and gathers local candidates. The resulting SDP answer is sent back via the signaling WebSocket.
- **GIL-Free Data Plane**: Once WebRTC connects natively, media packets flow entirely inside native Rust. Python is completely excluded from the media processing path.

---

## 7. Security & Authentication
- **Token Generation**: Access tokens are generated locally in Python using `livekit.api.AccessToken` signed with the `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`.
- **Credential Safety**: Credentials are read dynamically from environment variables (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`) and are never hardcoded.

---

## 8. Lifecycle & Thread Safety
- **Keepalives**: The signaling thread sends periodic Protobuf ping requests to prevent connection timeouts.
- **Graceful Shutdown**: Calling `stream.close()` cancels the background signaling task, closes the signaling WebSocket, terminates native Rust connection loops, and cleanly deallocates NVDEC decoder contexts.

---

## 9. Error Handling
- **Protobuf/Socket Failures**: WebSocket connection drops or signaling parse errors are propagated back to the main thread, raising a descriptive `RuntimeError`.
- **WebRTC/ICE Failures**: Handshake or negotiation errors in `webrtc-rs` propagate as `VeloConnectionError` exceptions.

---

## 10. Verification Status
- **Velo Native Media Plane**: **VERIFIED**
- **E2E WebSocket Signaling Protocol**: **EXPERIMENTALLY VERIFIED**
- **E2E LiveKit Server Direct Connection**: **UNVERIFIED / BLOCKED** (due to missing `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` credentials in the test execution environment).
