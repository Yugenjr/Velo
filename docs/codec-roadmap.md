# Velo Codec Roadmap

This document outlines the evaluation and priorities for expanding Velo's supported hardware-accelerated video codecs.

---

## 1. Codec Evaluation Matrix

| Codec | WebRTC Relevance | Browser Support | NVDEC Support | PyNvVideoCodec Support | Priority | Notes |
|---|---|---|---|---|---|---|
| **H.264** | **Critical** (Standard Baseline) | Universal | Universal (Turing+) | Yes | **Verified** | Current Velo baseline. Widely used for low-complexity WebRTC streaming. |
| **VP8** | **High** (Legacy Browser Standard) | Universal | Universal (Turing+) | Yes | **Medium** | Legacy WebRTC codec. Lower compression efficiency compared to H.264. |
| **VP9** | **High** (WebRTC Default) | Universal | Universal (Turing+) | Yes | **High** | Default WebRTC codec in Chrome/Safari for high-resolution video. Excellent compression. |
| **AV1** | **Emerging** (Next Gen Standard) | Chrome/Safari (Recent) | Ada Lovelace+ (RTX 40+) | Yes | **Low** | Next-gen standard. High compression but requires recent NVIDIA hardware for decode. |
| **HEVC/H.265** | **Low** | Safari only (WebRTC) | Universal (Turing+) | Yes | **Low** | Rarely supported natively in WebRTC by major browsers due to licensing, though standard in RTSP. |

---

## 2. Implementation Roadmap

### Phase 1: H.264 (Current - Verified)
The H.264 pipeline is fully implemented, verified, and performance-tested. It relies on standard Annex-B depacketization and `webrtc::rtp_transceiver` payload negotiations.

### Phase 2: VP9 (Planned)
- **Rationale**: VP9 is the default choice for modern high-definition browser-to-browser WebRTC streams (offering ~30% better compression than H.264).
- **Complexity**: Low to Moderate. Requires updating `MediaEngine` registrations in `_velo_native` and integrating VP9 RTP depacketization in Rust (`rtp::codecs::vp9`). The PyNvVideoCodec layer already supports VP9 decoding.
- **Impact**: Highly expands browser compatibility without sacrificing performance.

### Phase 3: VP8 (Planned)
- **Rationale**: Necessary to support legacy browsers and clients that do not support H.264 or VP9.
- **Complexity**: Low. Same structure as VP9 using `rtp::codecs::vp8` depacketizer.
- **Impact**: Broadens general compatibility.

### Phase 4: AV1 (Planned)
- **Rationale**: Next-generation standard with extreme compression efficiency, increasingly adopted by SFUs and LiveKit.
- **Complexity**: Moderate. Requires Ada Lovelace (RTX 40-series) or newer GPUs for hardware decoding. Dependency constraints block edge devices (like Jetson Nano/Orin).
- **Impact**: High performance on modern desktop/server hardware.
