# Velo Vision

## 1. Problem
Realtime multimodal AI applications can require continuous video/audio processing. Existing systems often require developers to combine multiple media, GPU, and AI technologies manually.

## 2. Proposed direction
Velo will explore a simple developer-facing abstraction:

Realtime media
    ↓
GPU-native processing
    ↓
GPU-resident frame/tensor
    ↓
AI inference

## 3. Initial target
WebRTC video → GPU-native frame → PyTorch/VLM inference.

## 4. Long-term direction
Potential future capabilities include:
- RTSP
- cameras
- multiple inference backends
- multi-stream processing
- adaptive frame scheduling
- observability
- benchmarking

## 5. Technical honesty
Velo is built on existing technologies such as WebRTC, CUDA, NVIDIA hardware decoding, GPU memory APIs, and AI frameworks. The intended value is in integration, abstraction, developer experience, and measurable system performance.

## 6. Development philosophy
Learn → Design → Implement → Test → Benchmark → Understand → Commit.
