import time
import threading
import pytest
import torch
import velo
from velo import (
    AIPipeline,
    AIScheduler,
    AudioScheduler,
    TemporalFusion,
    MockASRAdapter,
    MockMultimodalAdapter,
    RuntimeMetrics,
    RollingStats,
    Transcript,
    StreamClosedError,
)


class MockMultimodalStream:
    """Simulates a live Stream producing synchronized audio and video frames."""
    def __init__(self, fps=30, sample_rate=48000, channels=2, duration_s=1.0):
        self.fps = fps
        self.sample_rate = sample_rate
        self.channels = channels
        self.total_frames = int(fps * duration_s)
        self.total_audio_chunks = int(duration_s / 0.02)
        
        self.frame_idx = 0
        self.audio_idx = 0
        self.closed = False

    def next_frame(self):
        if self.closed or self.frame_idx >= self.total_frames:
            raise StreamClosedError("End of video stream")
        ts = self.frame_idx / self.fps
        self.frame_idx += 1
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tensor = torch.zeros((480, 640, 4), dtype=torch.uint8, device=device)
        
        class MockCapsule:
            def __init__(self, t, timestamp):
                self._t = t
                self.timestamp = timestamp
                self.shape = (480, 640, 4)
                self.__dlpack__ = t.__dlpack__
                self.__dlpack_device__ = t.__dlpack_device__
                
        return MockCapsule(tensor, ts), None

    def next_audio(self):
        if self.closed or self.audio_idx >= self.total_audio_chunks:
            raise StreamClosedError("End of audio stream")
        ts = self.audio_idx * 0.02
        self.audio_idx += 1
        
        num_samples = int(0.02 * self.sample_rate * self.channels)
        pcm = (torch.sin(torch.linspace(0, 100, num_samples)) * 16000).to(torch.int16)
        raw_bytes = pcm.numpy().tobytes()
        return raw_bytes, ts, self.channels, self.sample_rate, 0.02

    def close(self):
        self.closed = True


def test_rolling_stats_basic():
    stats = RollingStats(maxlen=10)
    # Empty snapshot
    s = stats.snapshot()
    assert s["count"] == 0
    assert s["mean"] == 0.0

    # Add 10 samples: 1.0 to 10.0
    for i in range(1, 11):
        stats.update(float(i))

    s = stats.snapshot()
    assert s["count"] == 10
    assert s["window_size"] == 10
    assert s["min"] == 1.0
    assert s["max"] == 10.0
    assert s["mean"] == 5.5
    assert s["median"] == 6.0
    assert s["latest"] == 10.0
    assert s["p95"] >= 9.0


def test_rolling_stats_bounded_window():
    stats = RollingStats(maxlen=5)
    # Add 100 samples
    for i in range(1, 101):
        stats.update(float(i))

    s = stats.snapshot()
    assert s["count"] == 100
    assert s["window_size"] == 5
    # Window should contain 96, 97, 98, 99, 100
    assert s["min"] == 96.0
    assert s["max"] == 100.0
    assert s["latest"] == 100.0
    assert s["mean"] == 98.0


def test_rolling_stats_thread_safety():
    stats = RollingStats(maxlen=500)

    def worker(start):
        for i in range(100):
            stats.update(float(start + i))

    threads = [threading.Thread(target=worker, args=(i * 100,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    s = stats.snapshot()
    assert s["count"] == 500
    assert s["window_size"] == 500


def test_runtime_metrics_direct():
    metrics = RuntimeMetrics()
    metrics.record_inference(inference_latency_ms=25.0, preprocessing_latency_ms=2.0, end_to_end_latency_ms=30.0)
    metrics.record_fusion(lookup_latency_ms=0.5, skew_ms=12.0)
    metrics.record_worker_error()
    metrics.record_inference_error()

    snap = metrics.snapshot(pipeline_state="RUNNING", inference_queue_depth=3)
    assert snap["system"]["pipeline_state"] == "RUNNING"
    assert snap["system"]["worker_errors"] == 1
    assert snap["inference"]["inference_count"] == 1
    assert snap["inference"]["inference_errors"] == 1
    assert snap["inference"]["inference_queue_depth"] == 3
    assert snap["inference"]["inference_latency_ms"]["latest"] == 25.0
    assert snap["inference"]["preprocessing_latency_ms"]["latest"] == 2.0
    assert snap["inference"]["end_to_end_latency_ms"]["latest"] == 30.0
    assert snap["fusion"]["fusion_queries"] == 1
    assert snap["fusion"]["lookup_latency_ms"]["latest"] == 0.5
    assert snap["fusion"]["timestamp_skew_ms"]["latest"] == 12.0


def test_temporal_fusion_stats():
    fusion = TemporalFusion(max_history_s=10.0)

    # Add mock frames and audio
    class MockCap:
        def __init__(self, ts):
            self.timestamp = ts
            self.shape = (10, 10, 3)
    f1 = velo.Frame(MockCap(1.0), None)
    f2 = velo.Frame(MockCap(1.5), None)

    fusion.add_video(f1, timestamp=1.0)
    fusion.add_video(f2, timestamp=1.5)

    t1 = Transcript(text="hello", start_timestamp=1.1, end_timestamp=1.4, confidence=0.9)
    fusion.add_audio(t1, timestamp=1.1, duration=0.3)

    # Query context at 1.2
    ctx = fusion.context(timestamp=1.2, window=0.5)
    assert len(ctx.video) == 2
    assert len(ctx.audio) == 1

    stats = fusion.stats()
    assert stats["queries_count"] == 1
    assert stats["video_buffer_size"] == 2
    assert stats["audio_buffer_size"] == 1
    assert stats["latest_skew_ms"] >= 0.0
    assert stats["latest_lookup_latency_ms"] >= 0.0


@pytest.mark.asyncio
async def test_pipeline_metrics_end_to_end():
    raw_stream = MockMultimodalStream(fps=30, sample_rate=48000, channels=2, duration_s=0.5)
    stream = velo.Stream(raw_stream)
    scheduler = AIScheduler(target_fps=15)
    audio_scheduler = AudioScheduler(chunk_duration_s=0.08, sample_rate=48000, channels=2)
    fusion = TemporalFusion(max_history_s=10.0)
    asr = MockASRAdapter(mock_text="observability query")
    model = MockMultimodalAdapter(simulated_latency=0.01)

    pipeline = AIPipeline(
        stream=stream,
        scheduler=scheduler,
        vlm=model,
        audio_scheduler=audio_scheduler,
        fusion=fusion,
        asr=asr,
        context_window_s=0.5,
    )

    await pipeline.start()

    # Query metrics while running
    m_running = pipeline.metrics()
    assert m_running["system"]["pipeline_state"] == "RUNNING"
    assert "video" in m_running
    assert "audio" in m_running
    assert "inference" in m_running
    assert "fusion" in m_running
    assert "gpu" in m_running

    async for _ in pipeline.run_inference("Metrics test"):
        pass

    await pipeline.stop()

    # Query metrics after completion
    m_done = pipeline.get_metrics()
    assert m_done["system"]["pipeline_state"] in ("STOPPED", "STOPPING")
    assert m_done["video"]["frames_received"] > 0
    assert m_done["audio"]["chunks_received"] > 0
    assert m_done["inference"]["inference_count"] > 0
    assert m_done["inference"]["inference_latency_ms"]["count"] > 0
    assert m_done["fusion"]["fusion_queries"] > 0
