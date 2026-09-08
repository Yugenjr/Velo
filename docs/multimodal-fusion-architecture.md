# Multimodal Fusion Architecture Audit & V1.7 Design

## 1. Audit Findings

1. **Existing timestamp representations:**
   Timestamps are represented as Python `float` values (seconds). 

2. **Video Frame Timestamps:**
   Yes, `velo.Frame` has a `timestamp` property derived from the native `_capsule.timestamp` (defaulting to 0.0 if unavailable).

3. **Audio Chunk Timestamps:**
   Yes, `AudioChunk` explicitly requires `timestamp` and `duration` as `float` attributes.

4. **Scheduler Snapshot Timestamps:**
   The `AIScheduler` preserves the `Frame` object (and thus its internal timestamp). However, its temporal window logic (`_evict_stale_frames` and `snapshot`) operates on the system `time.time()` of when the frame was *enqueued*, not the frame's actual capture timestamp.

5. **VLMResponse Representation:**
   `VLMResponse` only contains `text`, `latency_ms`, and `preprocessing_latency_ms`. It lacks temporal association with the frames used to generate it.

6. **AIPipeline Results:**
   `AIPipeline.run_inference()` is an async generator yielding raw `VLMResponse` objects. It does not return the originating frames or their timestamps.

7. **Audio/Video Lifecycle Coexistence:**
   They cannot currently coexist gracefully. `AIPipeline` exclusively manages video threads and state (`Stream`, `AIScheduler`, `VLMAdapter`). The audio components are completely separate and lack an orchestration pipeline.

8. **Accidental Multimodal Abstraction:**
   There are no accidental multimodal abstractions. The video pipeline and audio processing (tested in isolation) are entirely disjoint.

9. **Reusable Components:**
   - `Frame`, `AudioChunk`, and `Transcript` can be reused directly as observation payloads.
   - The proposed fusion layer can sit alongside these structures as an independent synchronization buffer without requiring modification to `Frame` or `Transcript`.

10. **Native Limitations Preventing Real Synchronization:**
    `native/src/lib.rs` explicitly drops non-H.264 tracks. Without real LiveKit audio ingestion, we lack real RTCP Sender Reports, meaning we cannot test true clock synchronization or handle network-induced clock drift between audio and video streams. All timestamps for testing must be simulated.

---

## 2. V1.7 Temporal Fusion Layer Architecture

### Core Purpose
The V1.7 Temporal Fusion layer strictly answers: *"Which audio/video observations belong together in time?"* It performs temporal grouping, windowing, and synchronization without any semantic reasoning.

### Proposed Abstractions

1. **`Observation` (Base)**
   - `timestamp: float`
   - `payload: Any`

2. **`VideoObservation` (extends Observation)**
   - `payload: Frame`

3. **`AudioObservation` (extends Observation)**
   - `payload: Transcript` (or `AudioChunk` if pre-ASR)
   - `duration: float`

4. **`MultimodalContext`**
   - `video: List[VideoObservation]`
   - `audio: List[AudioObservation]`
   - `reference_timestamp: float`
   
5. **`TemporalFusion` (The Core Buffer/Matcher)**
   - Thread-safe buffer for incoming observations.
   - **Configurable state:** `max_history_s` (memory bound), `sync_tolerance_s` (window tolerance).
   - **Methods:**
     - `add_video(frame: Frame, timestamp: float)`
     - `add_audio(transcript: Transcript, timestamp: float, duration: float)`
     - `context(timestamp: float, window: float) -> MultimodalContext` (Returns nearest matched context).
     - `latest_context(window: float) -> MultimodalContext`
   - **Behavior:** Automatically evicts stale observations to bound memory. Keeps GPU frames resident in the `VideoObservation` payload without forcing CPU conversion.

### Pipeline Integration Concept (For Future)
The `TemporalFusion` layer acts as a passive sink. The video pipeline deposits `Frame`s; the audio pipeline deposits `Transcript`s. The AI agent/VLM pulls `MultimodalContext`s from the fusion layer on demand.

### Implementation Risks
- **Testing:** We cannot test real synchronization drift due to the native WebRTC audio block.
- **VRAM Leaks:** Retaining `Frame` references in the temporal fusion history keeps the underlying CUDA allocations alive. The history window must be strictly bounded to prevent out-of-memory errors.
