import pytest
import asyncio
import time
import threading
import torch
from typing import List, Optional

import velo
from velo import (
    MultiStreamPipeline,
    StreamSession,
    MultiStreamResult,
    MockVLM,
    MockMultimodalAdapter,
    AIScheduler,
    AudioScheduler,
    AudioChunk,
    TemporalFusion,
    Frame,
    RuntimeMetrics,
)


class MockCapsule:
    def __init__(self, shape=(720, 1280, 3), timestamp=0.0):
        self.shape = shape
        self.timestamp = timestamp
        self._tensor = torch.zeros(shape, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.uint8)

    def __dlpack__(self, stream=None):
        return self._tensor.__dlpack__()

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


class MockStreamSource:
    def __init__(self, stream_id: str, fps: float = 30.0, total_frames: int = 60, has_audio: bool = False):
        self.stream_id = stream_id
        self.fps = fps
        self.total_frames = total_frames
        self.has_audio = has_audio
        self.frame_count = 0
        self.audio_count = 0
        self._closed = False
        self._lock = threading.Lock()

    def next(self) -> Optional[Frame]:
        with self._lock:
            if self._closed or self.frame_count >= self.total_frames:
                raise velo.StreamClosedError("End of mock stream")
            time.sleep(0.005)
            ts = self.frame_count * (1.0 / self.fps)
            capsule = MockCapsule(timestamp=ts)
            self.frame_count += 1
            frame = Frame(capsule, None)
            frame.arrival_time = time.time()
            return frame

    def next_audio(self) -> Optional[AudioChunk]:
        with self._lock:
            if self._closed or not self.has_audio or self.audio_count >= self.total_frames:
                raise velo.StreamClosedError("End of mock audio")
            time.sleep(0.005)
            ts = self.audio_count * 0.02
            samples = torch.zeros(960, dtype=torch.float32)
            self.audio_count += 1
            return AudioChunk(
                samples=samples,
                sample_rate=16000,
                channels=1,
                timestamp=ts,
                duration=0.02,
            )

    def close(self):
        with self._lock:
            self._closed = True


@pytest.mark.asyncio
async def test_multistream_1_stream():
    """Test 1 stream through MultiStreamPipeline."""
    vlm = MockVLM(simulated_latency=0.001)
    pipeline = MultiStreamPipeline(vlm=vlm)

    source = MockStreamSource("stream-1", fps=30.0, total_frames=15)
    session = pipeline.add_stream("stream-1", source, target_fps=15.0)

    await pipeline.start()
    results = []
    async for res in pipeline.run_inference("detect objects"):
        results.append(res)

    await pipeline.stop()

    assert len(results) > 0
    assert all(r.stream_id == "stream-1" for r in results)
    
    metrics = pipeline.metrics()
    s1_metrics = metrics.stream("stream-1")
    assert s1_metrics["inference"]["inference_count"] == len(results)
    assert metrics.global_snapshot()["system"]["active_streams"] == 1


@pytest.mark.asyncio
async def test_multistream_2_streams_fairness():
    """Test 2 streams with different framerates (30 FPS vs 15 FPS) for fair scheduling."""
    vlm = MockVLM(simulated_latency=0.001)
    pipeline = MultiStreamPipeline(vlm=vlm)

    source_a = MockStreamSource("stream-a", fps=30.0, total_frames=20)
    source_b = MockStreamSource("stream-b", fps=15.0, total_frames=10)

    pipeline.add_stream("stream-a", source_a, target_fps=30.0)
    pipeline.add_stream("stream-b", source_b, target_fps=15.0)

    await pipeline.start()
    stream_counts = {"stream-a": 0, "stream-b": 0}
    
    async for stream_id, response in pipeline.run_inference("analyze"):
        stream_counts[stream_id] += 1

    await pipeline.stop()

    assert stream_counts["stream-a"] > 0
    assert stream_counts["stream-b"] > 0
    # Both streams should have received fair inference allocations without starvation
    assert abs(stream_counts["stream-a"] - stream_counts["stream-b"]) < 15


@pytest.mark.asyncio
async def test_multistream_4_streams_multimodal():
    """Test 4 concurrent multimodal streams with audio and fusion."""
    adapter = MockMultimodalAdapter(simulated_latency=0.001)
    pipeline = MultiStreamPipeline(vlm=adapter)

    sources = []
    for i in range(4):
        sid = f"stream-{i}"
        src = MockStreamSource(sid, fps=30.0, total_frames=30, has_audio=True)
        sources.append(src)
        sched = AIScheduler(target_fps=10.0)
        audio_sched = AudioScheduler(chunk_duration_s=0.1)
        fusion = TemporalFusion(max_history_s=10.0)
        pipeline.add_stream(
            sid,
            src,
            scheduler=sched,
            audio_scheduler=audio_sched,
            fusion=fusion,
            context_window_s=1.0,
        )

    assert pipeline.active_stream_count == 4
    await pipeline.start()

    counts = {f"stream-{i}": 0 for i in range(4)}
    async for res in pipeline.run_inference("find anomalies"):
        counts[res.stream_id] += 1

    await pipeline.stop()

    for sid, count in counts.items():
        assert count > 0, f"Stream {sid} received 0 inference allocations!"

    g_metrics = pipeline.metrics().global_snapshot()
    assert g_metrics["inference"]["total_inferences"] == sum(counts.values())
    assert g_metrics["gpu"]["available"] is True


@pytest.mark.asyncio
async def test_multistream_8_streams():
    """Test 8 concurrent streams on available GPU hardware."""
    gpu_info = RuntimeMetrics.get_gpu_metrics()
    if gpu_info.get("available") and gpu_info.get("free_memory_mb", 0) < 150:
        pytest.skip("Insufficient GPU VRAM for 8 concurrent streams (<150 MB free)")

    vlm = MockVLM(simulated_latency=0.0005)
    pipeline = MultiStreamPipeline(vlm=vlm)

    sources = []
    for i in range(8):
        sid = f"cam-{i:02d}"
        src = MockStreamSource(sid, fps=20.0, total_frames=10)
        sources.append(src)
        pipeline.add_stream(sid, src, target_fps=10.0)

    assert pipeline.active_stream_count == 8
    await pipeline.start()

    counts = {f"cam-{i:02d}": 0 for i in range(8)}
    async for res in pipeline.run_inference("monitor"):
        counts[res.stream_id] += 1

    await pipeline.stop()

    # All 8 streams should complete inferences
    for sid, count in counts.items():
        assert count > 0, f"Stream {sid} was starved"
