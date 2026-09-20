"""
Velo Failure Injection & Resilience Test Suite (Milestone V1.12)

Verifies controlled failure injection, error propagation, resource release,
and thread teardown across all pipeline components.
"""
import time
import asyncio
import pytest
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
    PipelineState,
    AudioChunk,
    Frame,
)


class MockInfStream:
    """Stream that yields infinite valid frames and audio chunks without failing."""
    def __init__(self):
        self.frame_idx = 0
        self.audio_idx = 0
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._tensor = torch.zeros((100, 100, 3), dtype=torch.uint8, device=self.device)
        self._audio_bytes = (torch.full((960,), 10000, dtype=torch.int16)).numpy().tobytes()

    def next_frame(self):
        ts = self.frame_idx * 0.033
        self.frame_idx += 1
        t = self._tensor
        class MockCap:
            def __init__(self, t, ts):
                self.timestamp = ts
                self.shape = (100, 100, 3)
                self.__dlpack__ = t.__dlpack__
                self.__dlpack_device__ = t.__dlpack_device__
        return MockCap(t, ts), None

    def next_audio(self):
        ts = self.audio_idx * 0.02
        self.audio_idx += 1
        return self._audio_bytes, ts, 2, 48000, 0.02

    def close(self):
        pass


class FaultyVAD:
    """VAD that raises an error after N evaluations."""
    def __init__(self, fail_after=2):
        self.calls = 0
        self.fail_after = fail_after
    def analyze(self, chunk):
        self.calls += 1
        if self.calls >= self.fail_after:
            raise ValueError("Simulated VAD internal DSP computation fault")
        from velo.vad import VoiceActivityResult
        return VoiceActivityResult(is_speech=True, energy=0.5)


class FaultyASR:
    """ASR that raises an unrecoverable exception."""
    def __init__(self, fail_after=2):
        self.calls = 0
        self.fail_after = fail_after
    def transcribe(self, chunk):
        self.calls += 1
        if self.calls >= self.fail_after:
            raise RuntimeError("Simulated ASR model inference fault")
        return velo.Transcript(text="ok", start_timestamp=0.0, end_timestamp=0.2, confidence=0.9)


class FaultyVLM(velo.BaseMultimodalAdapter):
    """VLM that raises an exception during inference."""
    def __init__(self, fail_after=2):
        self.calls = 0
        self.fail_after = fail_after
    def generate(self, frames, prompt, **kwargs):
        self.calls += 1
        if self.calls >= self.fail_after:
            raise RuntimeError("Simulated GPU VLM Out-Of-Memory error")
        return velo.VLMResponse(text="valid", latency_ms=5.0)


def test_malformed_rtp_receiver():
    """Verify that corrupt/truncated RTP packets are rejected without crashing native runtime."""
    rx = velo.RtpReceiver(codec="h264", max_width=640, max_height=480)
    # 1. Empty payload
    rx.push_rtp(b"", 90000)
    # 2. Garbage random bytes
    rx.push_rtp(b"\xff\xff\x00\x12\x34\x56\x78", 93000)
    # 3. Invalid NAL header
    rx.push_rtp(b"\x00\x00\x00\x01\x1f\x99\x88", 96000)
    rx.close()


def test_audio_chunk_invalid_types():
    """Verify AudioChunk type enforcement."""
    with pytest.raises(TypeError):
        AudioChunk(samples="not_a_tensor", sample_rate=48000, channels=2, timestamp=0.0, duration=0.02)
    with pytest.raises(TypeError):
        AudioChunk(samples=torch.zeros(10), sample_rate="48000", channels=2, timestamp=0.0, duration=0.02)


@pytest.mark.asyncio
async def test_vad_failure_propagation():
    """Verify that a VAD failure cascades, stops the pipeline, and reports FAILED state."""
    raw_stream = MockInfStream()
    stream = velo.Stream(raw_stream)
    pipeline = AIPipeline(
        stream=stream,
        scheduler=AIScheduler(target_fps=30),
        vlm=MockMultimodalAdapter(),
        audio_scheduler=AudioScheduler(chunk_duration_s=0.1, sample_rate=48000, channels=2),
        fusion=TemporalFusion(),
        vad=FaultyVAD(fail_after=3),
    )
    
    await pipeline.start()
    with pytest.raises(ValueError, match="Simulated VAD internal DSP computation fault"):
        async for _ in pipeline.run_inference("test prompt"):
            await asyncio.sleep(0.02)
            
    assert pipeline.state == PipelineState.FAILED
    assert pipeline._metrics.worker_errors > 0
    await pipeline.stop()


@pytest.mark.asyncio
async def test_asr_failure_propagation():
    """Verify that an ASR failure cleanly cascades to the consumer."""
    raw_stream = MockInfStream()
    stream = velo.Stream(raw_stream)
    pipeline = AIPipeline(
        stream=stream,
        scheduler=AIScheduler(target_fps=30),
        vlm=MockMultimodalAdapter(),
        audio_scheduler=AudioScheduler(chunk_duration_s=0.1, sample_rate=48000, channels=2),
        fusion=TemporalFusion(),
        vad=VAD(energy_threshold=0.0), # always speech
        asr=FaultyASR(fail_after=2),
    )
    
    await pipeline.start()
    with pytest.raises(RuntimeError, match="Simulated ASR model inference fault"):
        async for _ in pipeline.run_inference("test prompt"):
            await asyncio.sleep(0.02)
            
    assert pipeline.state == PipelineState.FAILED
    await pipeline.stop()


@pytest.mark.asyncio
async def test_vlm_failure_propagation():
    """Verify that a VLM exception terminates the pipeline with proper metrics."""
    raw_stream = MockInfStream()
    stream = velo.Stream(raw_stream)
    pipeline = AIPipeline(
        stream=stream,
        scheduler=AIScheduler(target_fps=30),
        vlm=FaultyVLM(fail_after=3),
    )
    
    await pipeline.start()
    with pytest.raises(RuntimeError, match="Simulated GPU VLM Out-Of-Memory error"):
        async for _ in pipeline.run_inference("test prompt"):
            await asyncio.sleep(0.01)
            
    assert pipeline.state == PipelineState.FAILED
    assert pipeline._metrics.inference_errors > 0
    await pipeline.stop()


@pytest.mark.asyncio
async def test_shutdown_during_active_inference():
    """Verify that calling stop() during continuous inference cleanly stops workers."""
    raw_stream = MockInfStream()
    stream = velo.Stream(raw_stream)
    pipeline = AIPipeline(
        stream=stream,
        scheduler=AIScheduler(target_fps=30),
        vlm=MockMultimodalAdapter(simulated_latency=0.02),
    )
    
    await pipeline.start()
    
    async def stop_soon():
        await asyncio.sleep(0.1)
        await pipeline.stop()
        
    task = asyncio.create_task(stop_soon())
    
    try:
        async for _ in pipeline.run_inference("prompt"):
            pass
    except Exception:
        pass
        
    await task
    assert pipeline.state in (PipelineState.STOPPING, PipelineState.STOPPED)


@pytest.mark.asyncio
async def test_invalid_lifecycle_operations():
    """Verify guards against invalid state transitions."""
    raw_stream = MockInfStream()
    stream = velo.Stream(raw_stream)
    pipeline = AIPipeline(stream=stream, scheduler=AIScheduler(), vlm=MockMultimodalAdapter())
    
    # 1. run_inference before start() must fail
    with pytest.raises(RuntimeError, match="Pipeline must be RUNNING"):
        async for _ in pipeline.run_inference("prompt"):
            pass
            
    await pipeline.start()
    
    # 2. Duplicate start() must fail
    with pytest.raises(RuntimeError, match="Cannot start pipeline from state RUNNING"):
        await pipeline.start()
        
    await pipeline.stop()
