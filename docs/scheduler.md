# Velo AI-Aware Frame Scheduling

## Why an AI-Aware Scheduler?

In a typical WebRTC or CCTV pipeline, video frames arrive at 30 or 60 Frames Per Second (FPS). While a GPU hardware decoder (NVDEC) can easily process these frames in real-time without blocking the CPU, modern multimodal AI models (e.g., Vision-Language Models like Qwen-VL or LLaVA) are computationally expensive and typically run at much lower framerates—often between 1 and 5 FPS.

If Velo blindly handed every single 30 FPS decoded frame to a VLM, the AI queue would rapidly overflow, causing massive multi-second latency and ultimately OOM (Out-Of-Memory) crashes. 

**The AI-Aware Scheduler solves this impedance mismatch.**

## How it works

The `AIScheduler` acts as an elastic buffer between the high-speed WebRTC ingest and the low-speed AI inference loop.

1. **GPU-Native**: The scheduler operates strictly on object references (the `Frame` objects containing DLPack capsules). It does **not** copy the frame from GPU memory, nor does it convert it to CPU bytes.
2. **Latest-Frame Bias**: When the AI model asks for a frame, it usually wants to see what is happening *right now*. The scheduler uses a `latest` dropping strategy: if the bounded queue fills up, stale frames are aggressively dropped in favor of newer ones.
3. **Pacing and Backpressure**: The scheduler takes a `target_fps`. It paces the consumer loop to ensure the model isn't starved, but also isn't forced to spin unnecessarily.

## How Velo differs from DeepStream

NVIDIA DeepStream is an industry-standard video analytics SDK. It is designed for massive camera fleets and traditional object detection / tracking pipelines. DeepStream treats every frame equally and runs them through rigid metadata graphs (GStreamer). 

Velo, conversely, is an agile inference bridge. It does not treat video analytics as a fixed graph. Instead, Velo positions the **GPU-resident tensor** as the core abstraction. 

This enables dynamic, agentic interactions where an AI model might say:
> "Wait, I saw movement. Give me the next 5 frames at 30 FPS."

or:
> "Nothing is happening. Only wake me up at 1 FPS."

## Current Implementation

In this initial release, the `AIScheduler` provides deterministic scheduling:
- Fixed-rate target FPS sampling.
- Bounded buffering.
- Explicit stale-frame dropping on queue overflow.
- Thread-safe producer/consumer isolation.

## Future Roadmap

The scheduler is designed to be extensible. Future implementations will include:
- **Semantic Scene Detection**: Pre-processing modules that skip frames if the scene hasn't changed.
- **Adaptive Rates**: Dynamically slowing down or speeding up the scheduler based on visual activity.
- **Temporal Buffering**: Storing a rolling ring-buffer of the last 30 frames so the AI can request immediate historical context when an event occurs.
