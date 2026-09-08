# AIPipeline Orchestrator

The `AIPipeline` is Velo's high-level asynchronous orchestration layer. It bridges the gap between synchronous, zero-copy GPU inference streams and modern `asyncio`-based agentic ecosystems (such as LiveKit Agents or OpenAI Realtime APIs).

## Why AIPipeline?
Velo's core primitives (`Stream`, `AIScheduler`, `VLMAdapter`) are completely decoupled and operate synchronously to maximize GPU throughput and maintain zero-copy DLPack semantics.

However, modern multimodal agents are written in `asyncio`. Running blocking 30 FPS media ingestion and 5 FPS AI inference on the main asyncio event loop would stall the server.

The `AIPipeline` automatically manages these background threads for you.

## Architecture

```text
LiveKit / WebRTC
      ↓
(Ingest Worker Thread)
Stream.next()
      ↓
AIScheduler.submit()
      ↓
(Inference Worker Thread)
AIScheduler.acquire()  ─── gate ───> SceneChangeDetector
      ↓
(Candidate Gate) ─────── gate ───> CandidateSelector
      ↓
GPUPreprocessor
      ↓
VLMAdapter.generate()
      │
      ├── (Latency Feedback Loop) ──> AIScheduler.record_inference()
      ↓
(asyncio.Queue)
      ↓
run_inference() async generator
```

The pipeline automatically handles reporting the latency of the `VLMAdapter` back to the `AIScheduler`. If you construct the scheduler with `adaptive=True`, this latency feedback continuously tunes the inference frequency (target FPS) to match the real-time capabilities of your hardware without falling behind.

## Example Usage

```python
import asyncio
import velo

async def main():
    # Connect to the stream
    stream = velo.connect_livekit(
        url="ws://localhost:7880", 
        token="your_jwt_token"
    )

    # Construct the pipeline
    pipeline = velo.AIPipeline(
        stream=stream,
        scheduler=velo.AIScheduler(target_fps=5, scene_aware=True),
        vlm=velo.SmolVLMAdapter(use_gpu_preprocess=True)
    )

    # Start background threads
    await pipeline.start()

    print("Pipeline running...")

    try:
        # Consume inference results asynchronously
        async for result in pipeline.run_inference(prompt="Describe the scene"):
            print(f"[{result.latency_ms:.1f}ms]: {result.text}")
    finally:
        # Guarantee cleanup
        await pipeline.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## Lifecycle and Error Propagation

The pipeline has strict lifecycle states: `CREATED`, `STARTING`, `RUNNING`, `STOPPING`, `STOPPED`, and `FAILED`.

If an internal worker thread encounters an error (for example, the stream disconnects unexpectedly or the VLM crashes), the pipeline automatically transitions to the `FAILED` state.
The exception is caught and propagated cleanly into the `run_inference()` generator, raising it directly in your async consumer loop. This ensures you never have silent background thread crashes.

## Backpressure and GPU Memory
The `AIPipeline` relies entirely on the `AIScheduler` for backpressure. It does not unbounded-queue frames in memory.
If the VLM inference takes 5 seconds, the ingestion thread continues receiving WebRTC frames and submitting them to the scheduler, which silently drops stale frames while maintaining the temporal ring buffer. Zero `cpu().numpy()` conversion occurs.

## Advanced Usage
If you need absolute control over thread scheduling, multiple simultaneous VLMs, or custom ML hardware queues, do not use `AIPipeline`.
Instead, manually construct the `Stream` and `AIScheduler` and wire the loops yourself as described in the Core API documentation.

## Audio Constraints
> [!WARNING]
> The current `AIPipeline` is designed exclusively for **video** inference. Native WebRTC audio ingestion is currently unimplemented in the core C++/Rust backend. Velo V1.6 provides Python abstractions (`AudioScheduler`, `VAD`, `ASRAdapter`) in the `velo.audio` module for mock testing, but these are not yet wired into the `AIPipeline`.
