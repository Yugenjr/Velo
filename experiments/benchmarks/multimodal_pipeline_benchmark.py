"""
Velo End-to-End Multimodal Pipeline Benchmark

Benchmarks end-to-end throughput, audio/video ingestion synchronization,
fusion lookup overhead, inference latency distribution, and runtime metrics.
"""
import time
import sys
import os
import asyncio
import torch
import velo
from velo import (
    AIPipeline,
    AIScheduler,
    AudioScheduler,
    TemporalFusion,
    VAD,
    MockASRAdapter,
    MockMultimodalAdapter,
    StreamClosedError,
)
from benchmark_utils import save_benchmark_results


class BenchmarkStream:
    """Produces synchronized video (30 FPS) and audio (50 chunks/sec) for benchmark duration."""
    def __init__(self, fps=30, sample_rate=48000, channels=2, duration_s=15.0):
        self.fps = fps
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration_s = duration_s
        self.total_frames = int(fps * duration_s)
        self.total_audio_chunks = int(duration_s / 0.02)

        self.frame_idx = 0
        self.audio_idx = 0
        self.closed = False

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._tensor = torch.zeros((720, 1280, 3), dtype=torch.uint8, device=self.device)

        num_samples = int(0.02 * sample_rate * channels)
        pcm = (torch.sin(torch.linspace(0, 100, num_samples)) * 16000).to(torch.int16)
        self._audio_bytes = pcm.numpy().tobytes()

    def next_frame(self):
        if self.closed or self.frame_idx >= self.total_frames:
            raise StreamClosedError("End of video benchmark stream")
        time.sleep(1.0 / self.fps)
        ts = self.frame_idx / self.fps
        self.frame_idx += 1

        t = self._tensor
        class MockCapsule:
            def __init__(self, t, timestamp):
                self.timestamp = timestamp
                self.shape = (720, 1280, 3)
                self.__dlpack__ = t.__dlpack__
                self.__dlpack_device__ = t.__dlpack_device__

        return MockCapsule(t, ts), None

    def next_audio(self):
        if self.closed or self.audio_idx >= self.total_audio_chunks:
            raise StreamClosedError("End of audio benchmark stream")
        time.sleep(0.02)
        ts = self.audio_idx * 0.02
        self.audio_idx += 1
        return self._audio_bytes, ts, self.channels, self.sample_rate, 0.02

    def close(self):
        self.closed = True


async def run_multimodal_benchmark(duration_s=15.0, target_fps=15, simulated_inference_latency_ms=10.0):
    print(f"\n--- Running Multimodal Pipeline Benchmark ({duration_s}s, Target {target_fps} FPS, Model Latency {simulated_inference_latency_ms}ms) ---")
    raw_stream = BenchmarkStream(duration_s=duration_s)
    stream = velo.Stream(raw_stream)

    scheduler = AIScheduler(target_fps=target_fps, adaptive=True, max_temporal_frames=60)
    audio_scheduler = AudioScheduler(chunk_duration_s=0.2, sample_rate=48000, channels=2)
    fusion = TemporalFusion(max_history_s=30.0)
    vad = VAD(energy_threshold=0.01)
    asr = MockASRAdapter(mock_text="benchmark user speech")
    model = MockMultimodalAdapter(simulated_latency=simulated_inference_latency_ms / 1000.0)

    pipeline = AIPipeline(
        stream=stream,
        scheduler=scheduler,
        vlm=model,
        audio_scheduler=audio_scheduler,
        fusion=fusion,
        vad=vad,
        asr=asr,
        context_window_s=1.0,
    )

    t0 = time.time()
    await pipeline.start()

    inferences = 0
    async for response in pipeline.run_inference("Benchmark prompt for multimodal reasoning"):
        inferences += 1

    await pipeline.stop()
    t_elapsed = time.time() - t0

    metrics_snapshot = pipeline.metrics()

    inf_stats = metrics_snapshot["inference"]["inference_latency_ms"]
    e2e_stats = metrics_snapshot["inference"]["end_to_end_latency_ms"]
    fusion_stats = metrics_snapshot["fusion"]["lookup_latency_ms"]
    skew_stats = metrics_snapshot["fusion"]["timestamp_skew_ms"]
    gpu_stats = metrics_snapshot["gpu"]

    benchmark_summary = {
        "duration_s": round(duration_s, 2),
        "total_elapsed_s": round(t_elapsed, 3),
        "total_inferences": inferences,
        "effective_inference_fps": round(inferences / max(0.001, t_elapsed), 2),
        "video_frames_received": metrics_snapshot["video"]["frames_received"],
        "video_frames_dropped": metrics_snapshot["video"]["frames_dropped"],
        "audio_chunks_received": metrics_snapshot["audio"]["chunks_received"],
        "audio_chunks_dropped": metrics_snapshot["audio"]["chunks_dropped"],
        "inference_latency_p50_ms": inf_stats["median"],
        "inference_latency_p95_ms": inf_stats["p95"],
        "inference_latency_p99_ms": inf_stats["p99"],
        "end_to_end_latency_p50_ms": e2e_stats["median"],
        "end_to_end_latency_p95_ms": e2e_stats["p95"],
        "fusion_lookup_avg_ms": fusion_stats["mean"],
        "timestamp_skew_avg_ms": skew_stats["mean"],
        "timestamp_skew_max_ms": skew_stats["max"],
        "gpu_memory_allocated_mb": gpu_stats.get("memory_allocated_mb", 0.0),
    }

    print(f"\n[SUMMARY] Inferences: {inferences} ({benchmark_summary['effective_inference_fps']} FPS)")
    print(f"Video Ingested: {benchmark_summary['video_frames_received']} | Dropped: {benchmark_summary['video_frames_dropped']}")
    print(f"Audio Ingested: {benchmark_summary['audio_chunks_received']} | Dropped: {benchmark_summary['audio_chunks_dropped']}")
    print(f"Inference Latency: p50={inf_stats['median']}ms, p95={inf_stats['p95']}ms, p99={inf_stats['p99']}ms")
    print(f"End-to-End Latency: p50={e2e_stats['median']}ms, p95={e2e_stats['p95']}ms")
    print(f"Timestamp Skew: avg={skew_stats['mean']}ms, max={skew_stats['max']}ms")
    if gpu_stats.get("available"):
        print(f"GPU ({gpu_stats.get('device')}): Memory Allocated = {gpu_stats.get('memory_allocated_mb')} MB")

    config = {
        "duration_s": duration_s,
        "target_fps": target_fps,
        "simulated_inference_latency_ms": simulated_inference_latency_ms,
        "model_type": "MockMultimodalAdapter (simulated latency)",
    }
    save_benchmark_results("multimodal_pipeline", config, benchmark_summary)
    return benchmark_summary


if __name__ == "__main__":
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    asyncio.run(run_multimodal_benchmark(duration_s=dur))
