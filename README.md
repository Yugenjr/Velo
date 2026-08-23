# Velo

GPU-native realtime media for AI.

## What is Velo?
Velo is an experimental open-source project exploring a developer-friendly path from realtime media to GPU-resident AI workloads. 

## Why Velo?
Realtime multimodal AI applications often require continuous video and audio processing. Currently, developers must manually stitch together various media, GPU, and AI technologies. Velo aims to explore a clean abstraction to streamline this integration, making it easier to move realtime media directly into GPU-resident memory for AI inference.

## Current Status
Velo is in the very early stages of conceptualization and design. It is an experimental project; we have not implemented the system yet, nor have we established any performance benchmarks. 

## Long-Term Vision
The goal is to build a seamless pipeline:
Realtime media → GPU-native processing → GPU-resident frame/tensor → AI inference.

While the initial focus is on exploring WebRTC video to GPU-native frames for PyTorch/VLM inference, potential future capabilities may include RTSP support, camera ingestion, multi-stream processing, and adaptive frame scheduling.

## Design Principles
- **Developer Experience:** Provide a clean, intuitive abstraction.
- **Experimental Driven:** Learn → Design → Implement → Test → Benchmark → Understand → Commit.
- **Built on Giants:** Velo relies on existing, proven technologies like WebRTC, CUDA, NVIDIA hardware decoding, GPU memory APIs, and major AI frameworks.

## Roadmap
1. Architecture and technology validation.
2. Initial WebRTC to GPU frame pipeline exploration.
3. PyTorch/VLM inference integration experiments.
4. Benchmarking and telemetry implementation.

## License
Velo is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
