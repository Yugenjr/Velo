# Velo Audio Ingestion Architecture (V1.8)

## 1. Executive Summary

This document specifies the technical architecture for **V1.8 Real Native WebRTC / LiveKit Audio Ingestion** in Velo.
The goal is to seamlessly connect the native Rust WebRTC transport to the existing Python audio runtime (`AudioChunk`, `AudioScheduler`, `VAD`, `ASR`, `Transcript`) and multimodal temporal fusion (`TemporalFusion`).

```
                    LiveKit SFU / WebRTC
                             │
                             ├───────────────────────────────┐
                             │ [RTP H.264 @ 90kHz]           │ [RTP Opus @ 48kHz]
                             ▼                               ▼
                      Native Rust (Tokio)             Native Rust (Tokio)
                             │                               │
                      NVDEC HW Decode                 Opus C-API Decode
                             │                               │
                       [CUDA Frame]                    [PCM i16 Bytes]
                             │                               │
                       crossbeam (3)                  crossbeam (100)
                             │                               │
                             ▼                               ▼
                        Stream.next()                  Stream.next_audio()
                             │                               │
                             │ (torch.from_dlpack)           │ (torch.frombuffer)
                             ▼                               ▼
                       CUDA Frame                       AudioChunk
                             │                               │
                             ▼                               ▼
                        AIScheduler                    AudioScheduler
                             │                               │
                             ▼                               ▼
                        VLM / Agent                     VAD / ASR
                             │                               │
                             └───────────────┬───────────────┘
                                             │
                                             ▼
                                      TemporalFusion
```

---

## 2. Technical Audit & Architecture Analysis

### 2.1 Native WebRTC Layer (`webrtc-rs`)
- **Crate & Version:** `webrtc = "0.11.0"`.
- **Signaling & Negotiation:** Handles both H.264 video and Opus audio SDP codecs. `MediaEngine` is configured with `MIME_TYPE_OPUS` (payload type 111, clock rate 48,000 Hz, 2 channels).
- **Track Handling:** `pc.on_track(...)` detects incoming tracks via `track.codec().capability.mime_type`. When MIME type is `audio/opus`, it spawns a dedicated Tokio task.
- **Payload Decoding:** `webrtc-rs` yields raw RTP packets (`rtp_pkt.payload`). Opus payloads are decoded natively using `opus::Decoder::new(48000, Channels::Stereo)` (or Mono) into 16-bit linear PCM (`i16`).

### 2.2 Audio Representation & Format
- **Sample Format:** 16-bit signed integer linear PCM (`i16`), standard across WebRTC audio pipelines.
- **Native Sample Rate:** 48,000 Hz.
- **Channels:** 2 channels (Stereo) or 1 channel (Mono).
- **Frame Duration:** WebRTC Opus packets arrive in 10ms or 20ms chunks (960 stereo samples @ 20ms = 1920 `i16` = 3840 bytes).
- **Python Boundary:** Rust converts PCM slices to `PyBytes` and pairs them with an aligned relative timestamp in seconds: `(PyBytes, float)`. Python creates `AudioChunk` via zero-copy `torch.frombuffer(raw_bytes, dtype=torch.int16)`.

### 2.3 Timestamp Semantics & Multi-Clock Synchronization
- **Video Clock Domain:** RTP H.264 clock rate = 90,000 Hz.
- **Audio Clock Domain:** RTP Opus clock rate = 48,000 Hz.
- **Time Synchronization:**
  - When the first packet of a session arrives at wall-clock time $t_0$, the anchor timestamp is captured:
    - Audio anchor: $RTP_{audio, 0}$, $t_{audio, 0}$
    - Video anchor: $RTP_{video, 0}$, $t_{video, 0}$
  - For each subsequent audio packet with RTP timestamp $RTP_{audio}$:
    $$\text{timestamp}_{\text{audio}} = t_{audio, 0} + \frac{RTP_{audio} \ominus RTP_{audio, 0}}{48000}$$
  - For each video frame with RTP timestamp $RTP_{video}$:
    $$\text{timestamp}_{\text{video}} = t_{video, 0} + \frac{RTP_{video} \ominus RTP_{video, 0}}{90000}$$
  - Since $t_{audio, 0}$ and $t_{video, 0}$ are referenced to the exact same monotonic `Instant::now()` session start, both audio and video timestamps share a single unified monotonic time domain (seconds since stream start).
  - Wrapping arithmetic handles $2^{32}$ RTP timestamp rollover gracefully.

### 2.4 Native Threading & Memory Architecture
- **Dedicated Worker Thread:** `connect()` spawns an OS thread running Tokio multi-threaded runtime.
- **Non-blocking GIL Behavior:**
  - Audio and video decode tasks run concurrently inside Tokio.
  - Tokio tasks briefly acquire the Python GIL only when allocating Python objects (`PyBytes`, `PyTuple`) before pushing into lockless Crossbeam channels.
  - `Stream.next()` and `Stream.next_audio()` release the GIL (`py.allow_threads`) while waiting on their respective Crossbeam channels, preventing deadlocks and allowing concurrent execution.
- **Bounded Queues & Backpressure:**
  - `frame_rx`: Bounded to 3 items (drop-oldest policy).
  - `audio_rx`: Bounded to 100 items (~2.0 seconds of 20ms audio chunks, drop-oldest policy).
  - Audio memory footprint is strictly bounded (~384 KB max).

### 2.5 Lifecycle & Error Handling
- **Shutdown:** `stream.close()` sets `shutdown` atomic flag, unblocks Tokio loops, joins the worker thread with GIL released, and drains queues under the GIL.
- **Disconnections:** Remote WebRTC disconnection / track end cleanly terminates the Tokio loop and signals `StreamClosedError` to Python callers.

---

## 3. Python API Integration

### 3.1 `Stream` Additions
```python
class Stream:
    def next(self) -> Frame:
        """Block until next GPU video frame arrives."""
        ...

    def next_audio(self, timeout: Optional[float] = None) -> AudioChunk:
        """Block until next native AudioChunk arrives."""
        ...
```

### 3.2 `connect_livekit` Audio Ingestion
- Ingests both audio and video tracks over the single LiveKit WebRTC peer connection.
- Supports iterating audio via `stream.next_audio()` and submitting directly to `AudioScheduler(sample_rate=48000, channels=2)`.

---

## 4. Platform Considerations (Windows MSVC)
- Building `opus` C-bindings on Windows requires CMake policy configuration: `CMAKE_POLICY_VERSION_MINIMUM=3.5`.
- PyO3 0.21 on Python 3.13 requires `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` (or updating pyproject configuration).
