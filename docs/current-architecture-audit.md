# Velo Architecture Audit

## 1. Current Architecture Summary
Velo is a lightweight GPU-native realtime media-to-AI inference runtime designed primarily for multimodal AI applications. Its core video abstraction focuses on bypassing CPU-bound video decoding and memory copies by utilizing NVIDIA's hardware decoder (NVDEC) and directly feeding GPU-resident decoded frames to PyTorch-compatible tensors. 

The primary video path is:
`WebRTC (H.264 RTP) -> Rust Native Core -> NVDEC -> GPU Frame (DLPack) -> Temporal Buffer -> SceneChangeDetector -> CandidateSelector -> Adaptive AIScheduler -> GPUPreprocessor -> VLM Adapter -> AIPipeline -> Async VLMResponse`

## 2. Implemented Features
The repository contains complete implementations for the following components:
- **Rust Native Extension (`velo._velo_native`)**: Handles WebRTC signaling, H.264 depacketization, and feeds PyNvVideoCodec for GPU decoding.
- **Python Video Abstractions**:
  - `Stream`, `Frame` (GPU-resident representation).
  - `AIScheduler` (Bounded temporal buffering, backpressure, target FPS).
  - `Adaptive AIScheduler` (EWMA-style latency feedback).
  - `SceneChangeDetector` (L1/MAE GPU-native luminance comparison).
  - `CandidateSelector` (Visual difference magnitude heuristics before VLM).
  - `GPUPreprocessor` (On-GPU layout conversion, resize, normalization).
  - `BaseVLMAdapter` and `AIPipeline` (Asynchronous pipeline orchestrator handling threading/states).
- **Audio Abstractions (V1.6)**: 
  - `AudioChunk`, `AudioScheduler`, `VAD` (RMS/energy based), and `BaseASRAdapter` (with `MockASRAdapter`).

## 3. Verified Features
- Real local LiveKit E2E video (historically verified at ~29.62 FPS).
- GPU-native paths (zero-copy via DLPack) and NVDEC bindings.
- PyTorch CUDA availability and tensor mapping.
- Windows wheel build packaging (Maturin).

## 4. Blocked Features
- **Real VLM Inference**: Blocked by development machine constraints (4GB VRAM). The interface and mocked integrations are verified, but real large-model inference is hardware constrained.
- **Native WebRTC Audio Ingestion**: The native WebRTC transport in `native/src/lib.rs` explicitly filters out all tracks except H.264 video (`if codec != MIME_TYPE_H264.to_lowercase() { return; }`). Therefore, real LiveKit audio ingestion is blocked/unimplemented at the native layer.

## 5. Unimplemented Features
- **Multimodal Fusion**: Synchronization and semantic relations between audio and visual observations are not yet implemented.
- CPU fallback / non-NVDEC software decoding.
- 4K resolution stream verification.

## 6. Repository Discrepancies
- **Audio Runtime (V1.6)**: The prompt stated that V1.6 Audio Runtime was "what we were going to build next" and completely unimplemented. However, auditing the repository reveals that the audio abstractions (`audio.py`, `audio_scheduler.py`, `vad.py`, `asr.py`) and their corresponding tests (`test_audio.py`, `test_audio_scheduler.py`, `test_vad.py`, `test_asr.py`, `test_audio_pipeline.py`) **already exist** in the repository. 
- The V1.6 Audio Runtime is currently isolated; it is not yet fused into a unified `AIPipeline` with video, adhering to the requirement that V1.6 should not implement multimodal fusion yet.

## 7. Proposed V1.6 Audio Runtime Architecture
Since the code already exists, the proposed architecture matches the current implementation:
- **Audio Representation**: `AudioChunk` class representing samples, sample rate, channels, timestamps, and duration.
- **Audio Scheduler**: `AudioScheduler` enforcing bounded buffering, timestamp ordering, fixed chunk duration (e.g., 500ms), and backpressure.
- **VAD**: A lightweight deterministic heuristic `VAD` based on RMS energy, configurable with an energy threshold.
- **ASR**: A `BaseASRAdapter` interface and a `MockASRAdapter` yielding a `Transcript` with start/end timestamps and text.
- **Native Ingestion**: Audio ingestion from WebRTC remains blocked natively. We rely on the established mock infrastructure for testing.

## 8. Proposed Files to Create/Modify
- `src/velo/audio.py` (Already exists)
- `src/velo/audio_scheduler.py` (Already exists)
- `src/velo/vad.py` (Already exists)
- `src/velo/asr.py` (Already exists)
- `native/src/lib.rs` (To document that audio track ingestion is blocked)

## 9. Proposed Tests
The following tests have already been written and fulfill the requirements:
- `tests/test_audio.py` (Timestamps, ordering, chunking)
- `tests/test_audio_scheduler.py` (Bounded memory, backpressure, stats)
- `tests/test_vad.py` (Silence/speech detection, thresholds)
- `tests/test_asr.py` (ASR abstraction, transcript timestamps)
- `tests/test_audio_pipeline.py` (End-to-end simulated test of scheduler -> VAD -> ASR)

## 10. Implementation Risks
- **Clock Drift**: Simulating mock audio timestamps might mask clock drift issues that will appear when real WebRTC audio ingestion is implemented.
- **VAD Sensitivity**: The RMS-based VAD heavily depends on input normalization; it may struggle with varying microphone gain levels.
- **Future Integration**: Integrating the audio and video loops into a single asynchronous pipeline (multimodal fusion) will require careful thread synchronization to avoid GIL bottlenecks or blocking inference.
