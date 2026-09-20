# Velo AI Runtime Architecture

## 1. Current Architecture
Velo currently provides a powerful but low-level set of decoupled primitives. Data flows from a WebRTC signaling component into a C++/Rust native extension, decoding via NVDEC, and exposing a zero-copy PyTorch CUDA tensor. 

The architecture resembles a toolkit rather than an integrated runtime:
```text
[WebRTC Signaling / LiveKit]
       |
       +---------------- Video ----------------+---------------- Audio (V1.8) ---------------+
       |                                       |                                             |
       v                                       v                                             v
Native NVDEC -> Frame (CUDA)            Native Opus -> AudioChunk (CPU)                 AudioChunk
       |                                       |                                             |
       v                                       v                                             v
  AIScheduler                            AudioScheduler                               AudioScheduler
       |                                       |                                             |
       v                                       v                                             v
 SceneChangeDetector                          VAD                                           VAD
       |                                       |                                             |
 GPUPreprocessor                          ASRAdapter                                    ASRAdapter
       |                                       |                                             |
  VLM Adapter                             Transcript                                    Transcript
       |                                       |                                             |
       +---------------------------------------+---------------------------------------------+
                                               |
                                               v
                                        TemporalFusion
```

- **`Stream`**: Owns WebRTC signaling, RTP packet ingestion, NVDEC decoding, and native Opus audio decoding. Yields GPU `Frame` and `AudioChunk` objects.
- **`Frame`**: An RAII wrapper around a DLPack capsule. Owns GPU memory lifecycle.
- **`AudioChunk`**: Immutable tensor representation of an audio buffer, produced directly by native Opus decoding.
- **`AIScheduler`**: Maintains a temporal ring buffer. Handles FPS pacing and backpressure via explicit `submit()`, `acquire()`, and `release()`.
- **`AudioScheduler`**: Maintains a bounded buffer of `AudioChunk`s, applying backpressure and yielding consistent temporal slices (e.g., 500ms).
- **`VAD`**: Voice Activity Detector, an RMS energy heuristic.
- **`ASRAdapter`**: Interface for transcribing speech to text.
- **`SceneChangeDetector`**: A cheap GPU heuristic invoked internally by the scheduler to skip visually redundant frames.
- **`CandidateSelector`**: Evaluates whether a visually changed frame is worth spending VLM compute on, rejecting low-info frames based on spatial variance and cooldowns.
- **`GPUPreprocessor`**: Transforms `Frame`s into VLM-ready NCHW normalized tensors.
- **`VLMAdapter`**: Wraps Hugging Face model execution.

## 3. Identified Architectural Problems
1. **Thread Management Burden**: `Stream.next()` and `AIScheduler.acquire()` are blocking, synchronous methods. To maintain 30 FPS ingestion without blocking the 1-5 FPS AI inference, the developer is forced to manually construct and synchronize at least two OS threads.
2. **Lifecycle & Error Fragility**: If `stream.next()` raises a `StreamClosedError` (e.g., network drop), the ingestion thread dies. The developer must manually catch this and call `scheduler.close()` to wake up the inference thread waiting on `acquire()`, otherwise it deadlocks. 
3. **Async Ecosystem Gap**: Multimodal agents (using LiveKit, OpenAI, etc.) are heavily `asyncio` based. Velo forces users to bridge synchronous threading queues into async generators manually using `asyncio.to_thread()`.
4. **Violated Separation of Concerns (in proposed API)**: The conceptual API `connect_livekit(url, scheduler=..., model=...)` mixes WebRTC networking setup with AI model execution.

## 4. Proposed API Alternatives

### Alternative A: `connect_livekit` owns everything (The conceptual prompt)
```python
ai_stream = velo.connect_livekit(url, scheduler=AIScheduler(), model=VLMAdapter())
```
- *Pros*: Extremely concise.
- *Cons*: Highly coupled. `connect_livekit` shouldn't know about Hugging Face models. Difficult to support multiple VLMs (e.g., running a fast model at 5 FPS and a slow model at 1 FPS) or taking passive snapshots without a model.

### Alternative B: Integrated `AIStream` object
```python
stream = velo.connect_livekit(url)
ai_stream = velo.AIStream(stream, scheduler=AIScheduler(), model=VLMAdapter())
```
- *Pros*: Better separation. `AIStream` encapsulates the dual-thread ingestion/inference loop and exposes an async generator.
- *Cons*: Couples the scheduler tightly to the model execution.

### Alternative C: Asynchronous `AIPipeline` (Recommended)
Keep components decoupled but provide a high-level `AIPipeline` orchestrator that explicitly manages the threading, lifecycle cascading, and exposes a clean `async generator`.

## 5. Recommended Architecture (IMPLEMENTED)
**Alternative C: Asynchronous AIPipeline**

```python
# 1. Network / Decoding
stream = velo.connect_livekit(url)

# 2. Orchestration
pipeline = velo.AIPipeline(
    stream=stream,
    scheduler=velo.AIScheduler(target_fps=5, scene_aware=True),
    vlm=velo.SmolVLMAdapter(use_gpu_preprocess=True)
)

# 3. Async Consumption
async for result in pipeline.run_inference(prompt="What is happening?"):
    print(result.text, result.latency)
```

## 6. Proposed Lifecycle / State Machine
1. **`__enter__` or `.start()`**: `AIPipeline` spawns an internal *Ingestion Thread* and an *Inference Thread*.
2. **Ingestion Thread**: Loops `stream.next()` and feeds `scheduler.submit()`. 
3. **Inference Thread**: Loops `scheduler.acquire()`, calls `vlm.generate()`, queues the result to an `asyncio.Queue`, and calls `scheduler.release()`.
4. **Latency Feedback Loop**: The inference thread calculates the total latency of `vlm.generate()` and passes it to `scheduler.record_inference()`. If the scheduler is in adaptive mode (`adaptive=True`), this drives the target FPS dynamically via EWMA to match real-time hardware capabilities.
5. **Cascading Shutdown**: If `stream.next()` throws `StreamClosedError`, the ingestion thread catches it, calls `scheduler.close()`, which safely wakes the inference thread with `SchedulerClosedError`, which then puts a Sentinel/EOF on the async queue.

## 7. GPU Memory Ownership Model
- **`Stream`** allocates GPU memory via NVDEC.
- **`Frame`** holds the pointer.
- **`AIScheduler`** holds `Frame` references in its temporal `deque`.
- When frames fall out of the `temporal_window`, Python garbage collection drops the `Frame`, which decrements the DLPack refcount, freeing the GPU surface.

## 8. Backpressure Model
Remains entirely identical to V1.1. The `AIPipeline`'s internal inference thread naturally provides backpressure because it blocks on `vlm.generate()`. The ingestion thread continues at 30 FPS, and the scheduler efficiently drops old frames.

## 9. Async / Threading Model
The user interacts entirely in the `asyncio` domain. The heavy blocking operations (NVDEC polling and PyTorch inference) are sandboxed in dedicated OS threads managed transparently by the `AIPipeline`. 

## 10. Migration / Backward Compatibility
- Existing synchronous APIs (`Stream`, `AIScheduler`, `VLMAdapter`) remain completely untouched. 
- Advanced users can still wire them manually.
- The `AIPipeline` is simply a new top-level utility class in `src/velo/pipeline.py`.

## 11. What Should Explicitly NOT Be Abstracted
- **Model specific logic**: Do not bake Hugging Face or OpenAI specific tokenization into the Pipeline. The Pipeline should only expect a `VLMAdapter` interface (`.generate(frames, prompt)`).
- **WebRTC details**: Do not put SDP negotiation or WebSocket connection logic inside the Pipeline.

## 13. Runtime Observability & Metrics (V1.10)

Velo V1.10 introduces a zero-overhead, thread-safe runtime metrics and observability layer:

```text
AIPipeline / Runtime
   ├── Video: frames_received, frames_scheduled, frames_dropped, current_fps, average_fps, queue_depth
   ├── Audio: chunks_received, chunks_dropped, current_audio_rate, queue_depth
   ├── Inference: latency (mean/p50/p95/p99/min/max), preprocessing latency, end-to-end latency, errors
   ├── Fusion: queries, avg lookup latency, matched video/audio counts, timestamp skew (mean/max)
   ├── System: pipeline_state, worker_errors, uptime_seconds
   └── GPU: allocated_mb, reserved_mb, max_allocated_mb, device_name (safe query)
```

### Public Metrics API
```python
# Snapshot live runtime telemetry while pipeline is running
metrics = pipeline.metrics()  # or pipeline.get_metrics()

print(f"End-to-End Latency p95: {metrics['inference']['end_to_end_latency_ms']['p95']} ms")
print(f"Timestamp Skew Avg: {metrics['fusion']['timestamp_skew_ms']['mean']} ms")
print(f"Video Drops: {metrics['video']['frames_dropped']}")
```

### Reproducible Benchmarks
Benchmark suites are located in `experiments/benchmarks/`:
- `video_ingestion_benchmark.py`
- `audio_ingestion_benchmark.py`
- `scheduler_benchmark.py`
- `fusion_benchmark.py`
- `multimodal_pipeline_benchmark.py`

## 14. Real Hardware & Real Media Benchmarks (V1.11)

Milestone V1.11 establishes a verified baseline using real media and genuine GPU hardware decoding:

- **H.264 Hardware NVDEC**:
  - 480p: **3,486.5 ± 217.7 FPS**
  - 720p: **1,695.1 ± 10.5 FPS**
  - 1080p: **627.9 ± 186.8 FPS**
  - 4K UHD: **121.1 ± 27.4 FPS**
- **Zero-Copy DLPack Tensor Wrapping**: **9.9 to 10.5 $\mu\text{s}$**
- **VAD Processing Latency**: $p_{50} = \mathbf{269.6\, \mu\text{s}}$
- **Temporal Context Retrieval**: $p_{50} = \mathbf{3.67\, \mu\text{s}}$
- **Master Benchmark Harness**: `experiments/benchmarks/run_v1_11_suite.py` (3-run repeatability with mean and stddev)


