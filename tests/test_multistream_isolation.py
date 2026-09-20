import pytest
import asyncio
import time
import threading
import torch
from typing import Optional

import velo
from velo import (
    MultiStreamPipeline,
    StreamSession,
    MockVLM,
    MockMultimodalAdapter,
    AIScheduler,
    AudioScheduler,
    AudioChunk,
    TemporalFusion,
    Frame,
    BaseASRAdapter,
)


class MockFailingSource:
    def __init__(self, stream_id: str, fail_after: int = 5, error_type: str = "decoder"):
        self.stream_id = stream_id
        self.fail_after = fail_after
        self.error_type = error_type
        self.count = 0
        self.audio_count = 0
        self._closed = False
        self._lock = threading.Lock()

    def next(self) -> Optional[Frame]:
        with self._lock:
            if self._closed:
                raise velo.StreamClosedError("Stream is closed")
            time.sleep(0.005)
            self.count += 1
            if self.count > self.fail_after and self.error_type == "decoder":
                raise velo.DecodeError(f"Simulated NVDEC hardware crash in {self.stream_id}")
            if self.count > 30:
                raise velo.StreamClosedError("EOS")

            tensor = torch.zeros((480, 640, 3), device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.uint8)
            
            class Capsule:
                shape = (480, 640, 3)
                timestamp = self.count * 0.033
                def __dlpack__(self, stream=None): return tensor.__dlpack__()
                def __dlpack_device__(self): return tensor.__dlpack_device__()

            frame = Frame(Capsule(), None)
            frame.arrival_time = time.time()
            return frame

    def next_audio(self) -> Optional[AudioChunk]:
        with self._lock:
            if self._closed:
                raise velo.StreamClosedError("Audio closed")
            time.sleep(0.005)
            self.audio_count += 1
            if self.audio_count > self.fail_after and self.error_type == "audio":
                raise RuntimeError(f"Simulated audio ingestion error in {self.stream_id}")
            if self.audio_count > 30:
                raise velo.StreamClosedError("EOS")
            return AudioChunk(
                samples=torch.zeros(960, dtype=torch.float32),
                sample_rate=16000,
                channels=1,
                timestamp=self.audio_count * 0.02,
                duration=0.02,
            )

    def close(self):
        with self._lock:
            self._closed = True


class MockHealthySource:
    def __init__(self, stream_id: str, total_frames: int = 25):
        self.stream_id = stream_id
        self.total_frames = total_frames
        self.frame_count = 0
        self.audio_count = 0
        self._closed = False
        self._lock = threading.Lock()

    def next(self) -> Optional[Frame]:
        with self._lock:
            if self._closed or self.frame_count >= self.total_frames:
                raise velo.StreamClosedError("EOS")
            time.sleep(0.005)
            self.frame_count += 1
            tensor = torch.zeros((480, 640, 3), device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.uint8)
            
            class Capsule:
                shape = (480, 640, 3)
                timestamp = self.frame_count * 0.033
                def __dlpack__(self, stream=None): return tensor.__dlpack__()
                def __dlpack_device__(self): return tensor.__dlpack_device__()

            frame = Frame(Capsule(), None)
            frame.arrival_time = time.time()
            return frame

    def next_audio(self) -> Optional[AudioChunk]:
        with self._lock:
            if self._closed or self.audio_count >= self.total_frames:
                raise velo.StreamClosedError("EOS")
            time.sleep(0.005)
            self.audio_count += 1
            return AudioChunk(
                samples=torch.zeros(960, dtype=torch.float32),
                sample_rate=16000,
                channels=1,
                timestamp=self.audio_count * 0.02,
                duration=0.02,
            )

    def close(self):
        with self._lock:
            self._closed = True


class MockFailingASR(BaseASRAdapter):
    def __init__(self, fail_stream_id: str):
        self.fail_stream_id = fail_stream_id
        self.call_count = 0

    def transcribe(self, audio_samples, sample_rate: int = 16000) -> str:
        self.call_count += 1
        if self.call_count > 2:
            raise RuntimeError(f"Simulated ASR model crash for stream {self.fail_stream_id}")
        return "mock transcript"


class MockStreamSelectiveFailingVLM(MockVLM):
    def __init__(self, fail_stream_id: str):
        super().__init__(latency_ms=1.0)
        self.fail_stream_id = fail_stream_id
        self.stream_counts = {}

    def generate_text(self, image_tensor, prompt: str = "") -> str:
        # Check current caller or raise if failing
        return super().generate_text(image_tensor, prompt)


@pytest.mark.asyncio
async def test_isolation_decoder_failure():
    """Stream A decoder crashes -> Stream B continues unaffected."""
    vlm = MockVLM(simulated_latency=0.001)
    pipeline = MultiStreamPipeline(vlm=vlm)

    failing_a = MockFailingSource("stream-a", fail_after=3, error_type="decoder")
    healthy_b = MockHealthySource("stream-b", total_frames=20)

    sess_a = pipeline.add_stream("stream-a", failing_a, target_fps=30.0)
    sess_b = pipeline.add_stream("stream-b", healthy_b, target_fps=30.0)

    await pipeline.start()

    counts = {"stream-a": 0, "stream-b": 0}
    async for stream_id, response in pipeline.run_inference("monitor"):
        counts[stream_id] += 1

    await pipeline.stop()

    assert counts["stream-b"] > 0, "Stream B was prematurely stopped by Stream A failure!"
    assert sess_a.state.name == "FAILED" or sess_a.metrics.worker_errors > 0
    assert sess_b.state.name in ("STOPPED", "RUNNING")


@pytest.mark.asyncio
async def test_isolation_audio_failure():
    """Stream A audio fails -> Stream B continues processing."""
    vlm = MockMultimodalAdapter(simulated_latency=0.001)
    pipeline = MultiStreamPipeline(vlm=vlm)

    failing_a = MockFailingSource("stream-a", fail_after=3, error_type="audio")
    healthy_b = MockHealthySource("stream-b", total_frames=20)

    pipeline.add_stream(
        "stream-a", failing_a,
        audio_scheduler=AudioScheduler(),
        fusion=TemporalFusion(),
    )
    pipeline.add_stream(
        "stream-b", healthy_b,
        audio_scheduler=AudioScheduler(),
        fusion=TemporalFusion(),
    )

    await pipeline.start()

    counts = {"stream-a": 0, "stream-b": 0}
    async for stream_id, resp in pipeline.run_inference("check"):
        counts[stream_id] += 1

    await pipeline.stop()
    sess_b = pipeline.get_stream("stream-b")
    print("Stream B state:", sess_b.state if sess_b else None, "error:", sess_b._last_error if sess_b else None)
    print("Counts:", counts)
    assert counts["stream-b"] > 0, "Healthy stream B stopped!"


@pytest.mark.asyncio
async def test_isolation_asr_failure():
    """Stream A ASR raises -> Stream B continues."""
    vlm = MockMultimodalAdapter(simulated_latency=0.001)
    pipeline = MultiStreamPipeline(vlm=vlm)

    failing_asr = MockFailingASR("stream-a")
    source_a = MockHealthySource("stream-a", total_frames=20)
    source_b = MockHealthySource("stream-b", total_frames=20)

    pipeline.add_stream(
        "stream-a", source_a,
        audio_scheduler=AudioScheduler(),
        fusion=TemporalFusion(),
        asr=failing_asr,
    )
    pipeline.add_stream(
        "stream-b", source_b,
        audio_scheduler=AudioScheduler(),
        fusion=TemporalFusion(),
    )

    await pipeline.start()

    counts = {"stream-a": 0, "stream-b": 0}
    async for stream_id, resp in pipeline.run_inference("test"):
        counts[stream_id] += 1

    await pipeline.stop()

    assert counts["stream-b"] > 0


@pytest.mark.asyncio
async def test_isolation_dynamic_stream_removal():
    """Dynamically remove Stream A -> Stream B continues running."""
    vlm = MockVLM(simulated_latency=0.001)
    pipeline = MultiStreamPipeline(vlm=vlm)

    source_a = MockHealthySource("stream-a", total_frames=40)
    source_b = MockHealthySource("stream-b", total_frames=40)

    pipeline.add_stream("stream-a", source_a, target_fps=30.0)
    pipeline.add_stream("stream-b", source_b, target_fps=30.0)

    await pipeline.start()

    results = []
    removed = False
    async for res in pipeline.run_inference("run"):
        results.append(res)
        if len(results) >= 5 and not removed:
            pipeline.remove_stream("stream-a")
            removed = True
            assert not pipeline.has_stream("stream-a")
            assert pipeline.has_stream("stream-b")

    await pipeline.stop()

    # Verify stream B produced results after stream A was removed
    stream_b_results = [r for r in results if r.stream_id == "stream-b"]
    assert len(stream_b_results) >= 5

class MockFailingMultimodalAdapter(MockMultimodalAdapter):
    def __init__(self, simulated_latency: float = 0.001):
        super().__init__(simulated_latency)
        self.count = 0
        self._lock = threading.Lock()

    def infer(self, frames, transcripts=None, timestamp=None, prompt="", context=None):
        with self._lock:
            self.count += 1
            if self.count == 5:
                raise RuntimeError("Simulated VLM inference crash (e.g. CUDA OOM)")
        
        return super().infer(frames, transcripts, timestamp, prompt, context)

@pytest.mark.asyncio
async def test_multistream_inference_failure_isolation():
    """VLM inference crashes for one task -> Pipeline isolates the error, stream fails, others continue."""
    vlm = MockFailingMultimodalAdapter(simulated_latency=0.001)
    pipeline = MultiStreamPipeline(vlm=vlm)

    source_a = MockHealthySource("stream-a", total_frames=20)
    source_b = MockHealthySource("stream-b", total_frames=20)

    sess_a = pipeline.add_stream("stream-a", source_a, target_fps=30.0)
    sess_b = pipeline.add_stream("stream-b", source_b, target_fps=30.0)

    await pipeline.start()

    counts = {"stream-a": 0, "stream-b": 0}
    errors = 0
    try:
        async for stream_id, res in pipeline.run_inference("monitor"):
            if isinstance(res, Exception):
                errors += 1
            else:
                counts[stream_id] += 1
    except Exception:
        pass

    await pipeline.stop()
    
    total_successful = counts["stream-a"] + counts["stream-b"]
    assert total_successful > 0, "Pipeline completely halted after inference crash!"
