# Velo AI Runtime Architecture

## 1. Current Architecture
Velo currently provides a powerful but low-level set of decoupled primitives. Data flows from a WebRTC signaling component into a C++/Rust native extension, decoding via NVDEC, and exposing a zero-copy PyTorch CUDA tensor. 

The architecture resembles a toolkit rather than an integrated runtime:
```text
WebRTC Signaling -> Native Decoder -> Frame (CUDA) 
                                        | (Manual Thread)
                                        v
                                  AIScheduler
                                        | (Manual Thread)
                                        v
                               SceneChangeDetector
                                        |
                                  GPUPreprocessor
                                        |
                                  VLM Adapter
```

## 2. Current Component Responsibilities
- **`velo.connect_livekit` / `Stream`**: Owns WebRTC signaling, RTP packet ingestion, and NVDEC decoding. Yields `Frame` objects synchronously via `stream.next()`.
- **`Frame`**: An RAII wrapper around a DLPack capsule. Owns GPU memory lifecycle.
- **`AIScheduler`**: Maintains a temporal ring buffer. Handles FPS pacing and backpressure via explicit `submit()`, `acquire()`, and `release()`.
- **`SceneChangeDetector`**: A cheap GPU heuristic invoked internally by the scheduler to skip visually redundant frames.
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

## 12. Affected Files (For Future Implementation)
- **New File**: `src/velo/pipeline.py` (Implementation of `AIPipeline`).
- **Modifications**: `src/velo/__init__.py` (Export `AIPipeline`).
- **Tests**: `tests/test_pipeline.py` (New tests verifying async generation and thread teardown).
- **Docs**: `docs/getting-started.md` (Update tutorial to use `AIPipeline`).
