# Velo AI-Aware Temporal Scheduling

## Why an AI-Aware Temporal Scheduler?

In a typical WebRTC or CCTV pipeline, video frames arrive at 30 or 60 Frames Per Second (FPS). While a GPU hardware decoder (NVDEC) can easily process these frames in real-time without blocking the CPU, modern multimodal AI models (e.g., Vision-Language Models like Qwen-VL or LLaVA) are computationally expensive and typically run at much lower framerates—often between 1 and 5 FPS.

If Velo blindly handed every single 30 FPS decoded frame to a VLM, the AI queue would rapidly overflow, causing massive multi-second latency and ultimately OOM (Out-Of-Memory) crashes. 

**The AI-Aware Scheduler solves this impedance mismatch by providing a model-aware temporal buffer.**

## The Evolution of the Scheduler

### V1: Fixed-Rate / Latest-Frame Scheduling
Initially, the scheduler provided a basic bounded queue that enforced a `target_fps`. It dropped the oldest frame when the queue filled up, ensuring the consumer loop received the freshest frame without starvation.

### V1.1: Temporal Buffering + Model Backpressure
In V1.1, the scheduler was evolved into a powerful **temporal scheduling layer**:
1. **Temporal Buffer**: Instead of a simple `deque`, it maintains a ring-buffer bounded by `temporal_window` (seconds) and `max_temporal_frames`. This allows the AI to request historical context.
2. **Model Backpressure**: The API introduces an explicit `acquire()` / `release()` lifecycle.
   - `acquire()`: Blocks until a fresh frame arrives and marks the AI worker as *busy*.
   - `submit()`: While the model is *busy*, incoming 30 FPS frames are appended to the temporal buffer but are immediately marked as skipped (backpressure dropped) for direct inference.
   - `release()`: Marks the model as *idle* and ready for the next frame.
3. **GPU-Native Snapshots**: A VLM can call `snapshot(duration=1.0)` to instantly retrieve a chronologically ordered list of recent frames, perfectly preserving zero-copy GPU residency.

## The Data Flow

```text
WebRTC
  ↓
NVDEC
  ↓
GPU Frame
  ↓
AIScheduler
  ↓
VLM Adapter
```

## How Velo differs from DeepStream

NVIDIA DeepStream is an industry-standard video analytics SDK. It is designed for massive camera fleets and traditional object detection / tracking pipelines. DeepStream treats every frame equally and runs them through rigid metadata graphs (GStreamer). 

Velo, conversely, is an agile inference bridge. It does not treat video analytics as a fixed graph. Instead, Velo positions the **GPU-resident tensor** as the core abstraction. 

Its purpose is to make realtime media usable by multimodal AI models while minimizing unnecessary buffering, memory movement, and inference on stale frames.

This enables dynamic, agentic interactions where an AI model might say:
> "Wait, I saw movement. Give me a snapshot of the last 1.5 seconds to understand what happened."

or:
> "Nothing is happening. Only wake me up at 1 FPS."

## Scene-Aware Scheduling (V1.2)

To avoid redundant VLM inference on visually static frames, Velo introduces **Scene-Aware Scheduling**. This acts as a cheap GPU-native heuristic gatekeeper before expensive AI models are invoked.

*Note: This is a temporal visual-change heuristic, not a semantic understanding model. It only measures pixel/luminance shifts to reduce redundant work.*

```text
                    ┌── unchanged ──→ skip AI
                    │
Incoming Frame → Scheduler → Change Detector
                    │
                    └── changed ───→ VLM
                                      ↓
                                  AI Response
```

When `scene_aware=True`, the scheduler behaves exactly as before (buffering 30 FPS history in the temporal window), but the inference thread pulling from `.acquire()` will silently skip and wait for the next frame if the current frame is visually identical to the last analyzed frame.

**Why this differs from conventional video pipelines:**
The scheduler is optimizing for AI inference workload rather than simply maintaining a media pipeline. By accumulating subtle changes in a 64x64 luminance tensor, Velo drops unnecessary VLM requests while keeping the temporal buffer perfectly intact for historical context.

## Adaptive AI-Aware Scheduling (V1.4)

While early versions used a fixed `target_fps` to throttle inference, V1.4 introduces **Adaptive Scheduling**. Multimodal models have highly variable latencies depending on prompt complexity, image context, and background hardware contention. 

If the model is given a strict 10 FPS target but can only physically execute at 3 FPS, a fixed scheduler will struggle, and the system might fall behind real-time.

When `adaptive=True` is configured, the `AIScheduler` relies on an Exponentially Weighted Moving Average (EWMA) of the inference latency.

```python
scheduler = AIScheduler(
    target_fps=5.0,
    adaptive=True,
    min_fps=1.0,
    max_fps=30.0,
    latency_budget_ms=250.0
)
```

**How it works:**
1. The inference pipeline (e.g. `AIPipeline`) feeds real-time latency back to the scheduler via `scheduler.record_inference(latency_ms)`.
2. The scheduler updates its internal EWMA latency.
3. If the EWMA latency exceeds the configured `latency_budget_ms`, the scheduler automatically scales down the `target_fps` (by a factor, typically 0.9) to prevent falling behind.
4. If the model operates much faster than the budget (e.g., < 50% of the budget), the scheduler scales the `target_fps` up (by a factor, typically 1.05) to maximize utilization.
5. The dynamic `target_fps` is strictly clamped between `min_fps` and `max_fps`.

This guarantees bounded memory utilization, zero inference queue backlogs, and maximizes the capabilities of the specific hardware executing the VLM.

## Future Roadmap

The scheduler is designed to be extensible. Future implementations will include:
- **VLM-driven Frame Requests**: Allowing the AI agent to explicitly drive the ingestion loop based on conversational context.
