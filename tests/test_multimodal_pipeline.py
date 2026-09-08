import asyncio
import time
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
    BaseMultimodalAdapter,
    VLMResponse,
    Transcript,
    StreamClosedError,
    PipelineState,
)


class MockMultimodalStream:
    """Simulates a live Stream producing synchronized audio and video frames."""
    def __init__(self, fps=30, sample_rate=48000, channels=2, duration_s=1.0, is_cuda=True):
        self.fps = fps
        self.sample_rate = sample_rate
        self.channels = channels
        self.total_frames = int(fps * duration_s)
        self.total_audio_chunks = int(duration_s / 0.02) # 20ms chunks
        self.is_cuda = is_cuda
        
        self.frame_idx = 0
        self.audio_idx = 0
        self.closed = False

    def next_frame(self):
        if self.closed or self.frame_idx >= self.total_frames:
            raise StreamClosedError("End of video stream")
        ts = self.frame_idx / self.fps
        self.frame_idx += 1
        
        device = "cuda:0" if (self.is_cuda and torch.cuda.is_available()) else "cpu"
        # Simulate GPU tensor surface
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
        # Generate non-silent audio so VAD detects speech
        pcm = (torch.sin(torch.linspace(0, 100, num_samples)) * 16000).to(torch.int16)
        raw_bytes = pcm.numpy().tobytes()
        return raw_bytes, ts, self.channels, self.sample_rate, 0.02

    def close(self):
        self.closed = True


class SlowMultimodalAdapter(BaseMultimodalAdapter):
    """Simulates a slow multimodal model to test backpressure."""
    def __init__(self, latency_s: float = 0.1):
        self.latency_s = latency_s
        self.infer_count = 0

    def infer(self, frames, transcripts=None, timestamp=None, prompt="", context=None):
        time.sleep(self.latency_s)
        self.infer_count += 1
        text = f"Inferred frame at ts={timestamp} with {len(transcripts or [])} transcripts"
        return VLMResponse(
            text=text,
            latency_ms=self.latency_s * 1000,
            timestamp=timestamp,
            transcripts=transcripts,
            context=context,
        )


@pytest.mark.asyncio
async def test_multimodal_pipeline_end_to_end():
    """Verify that AIPipeline orchestrates video + audio + VAD + ASR + fusion + model."""
    raw_stream = MockMultimodalStream(fps=30, sample_rate=48000, channels=2, duration_s=0.5)
    stream = velo.Stream(raw_stream)
    scheduler = AIScheduler(target_fps=15)
    audio_scheduler = AudioScheduler(chunk_duration_s=0.08, sample_rate=48000, channels=2)
    fusion = TemporalFusion(max_history_s=10.0)
    vad = VAD(energy_threshold=0.01)
    asr = MockASRAdapter(mock_text="user speech at timestamp")
    model = MockMultimodalAdapter(simulated_latency=0.01)

    pipeline = AIPipeline(
        stream=stream,
        scheduler=scheduler,
        vlm=model,
        audio_scheduler=audio_scheduler,
        fusion=fusion,
        vad=vad,
        asr=asr,
        context_window_s=0.5,
    )

    await pipeline.start()
    assert pipeline.state == PipelineState.RUNNING

    results = []
    async for response in pipeline.run_inference("Describe what the person said and showed"):
        results.append(response)

    await pipeline.stop()
    assert pipeline.state == PipelineState.STOPPED

    assert len(results) > 0
    # Check that responses have multimodal attributes
    first_resp = results[0]
    assert isinstance(first_resp, VLMResponse)
    assert first_resp.timestamp is not None
    assert isinstance(first_resp.text, str)
    assert len(first_resp.text) > 0


@pytest.mark.asyncio
async def test_multimodal_gpu_residency_preserved():
    """Verify that video frames in MultimodalContext remain GPU-resident."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    raw_stream = MockMultimodalStream(fps=30, duration_s=0.3, is_cuda=True)
    stream = velo.Stream(raw_stream)
    scheduler = AIScheduler(target_fps=30)
    audio_scheduler = AudioScheduler(chunk_duration_s=0.05, sample_rate=48000, channels=2)
    fusion = TemporalFusion(max_history_s=5.0)

    class CudaCheckingAdapter(BaseMultimodalAdapter):
        def __init__(self):
            self.verified_cuda = False

        def infer(self, frames, transcripts=None, timestamp=None, prompt="", context=None):
            if not isinstance(frames, list):
                frames = [frames]
            for f in frames:
                t = f.to_torch()
                assert t.is_cuda, f"Expected CUDA tensor, got device: {t.device}"
                self.verified_cuda = True

            # Also check frames inside context
            if context and context.video:
                for v_obs in context.video:
                    vt = v_obs.payload.to_torch()
                    assert vt.is_cuda, "Context video frame is not on CUDA!"

            return VLMResponse(text="CUDA OK", timestamp=timestamp, transcripts=transcripts, context=context)

    adapter = CudaCheckingAdapter()
    pipeline = AIPipeline(
        stream=stream,
        scheduler=scheduler,
        vlm=adapter,
        audio_scheduler=audio_scheduler,
        fusion=fusion,
        asr=MockASRAdapter(),
    )

    await pipeline.start()
    async for _ in pipeline.run_inference("check cuda"):
        pass
    await pipeline.stop()

    assert adapter.verified_cuda, "Adapter never verified CUDA residency!"


@pytest.mark.asyncio
async def test_multimodal_backpressure_and_pacing():
    """Verify that slow multimodal inference paces gracefully without unconstrained queue growth."""
    raw_stream = MockMultimodalStream(fps=30, duration_s=0.6)
    stream = velo.Stream(raw_stream)
    scheduler = AIScheduler(target_fps=30)
    audio_scheduler = AudioScheduler(chunk_duration_s=0.05, sample_rate=48000, channels=2)
    fusion = TemporalFusion(max_history_s=5.0)
    slow_model = SlowMultimodalAdapter(latency_s=0.05)

    pipeline = AIPipeline(
        stream=stream,
        scheduler=scheduler,
        vlm=slow_model,
        audio_scheduler=audio_scheduler,
        fusion=fusion,
        asr=MockASRAdapter(),
    )

    await pipeline.start()
    count = 0
    async for resp in pipeline.run_inference("Pacing test"):
        count += 1

    await pipeline.stop()
    assert count > 0
    assert slow_model.infer_count == count


@pytest.mark.asyncio
async def test_multimodal_cascading_error_propagation():
    """Verify that a model failure cleanly stops all background threads and surfaces exception."""
    class FailingMultimodalAdapter(BaseMultimodalAdapter):
        def infer(self, frames, transcripts=None, timestamp=None, prompt="", context=None):
            raise RuntimeError("Multimodal model OOM / catastrophic failure")

    raw_stream = MockMultimodalStream(fps=30, duration_s=1.0)
    stream = velo.Stream(raw_stream)
    scheduler = AIScheduler(target_fps=30)
    audio_scheduler = AudioScheduler(chunk_duration_s=0.05, sample_rate=48000, channels=2)
    fusion = TemporalFusion()

    pipeline = AIPipeline(
        stream=stream,
        scheduler=scheduler,
        vlm=FailingMultimodalAdapter(),
        audio_scheduler=audio_scheduler,
        fusion=fusion,
    )

    await pipeline.start()

    with pytest.raises(RuntimeError, match="Multimodal model OOM"):
        async for _ in pipeline.run_inference("Fail test"):
            pass

    assert pipeline.state == PipelineState.FAILED

    # Clean shutdown
    await pipeline.stop()
    for t in (pipeline._ingest_thread, pipeline._audio_ingest_thread, pipeline._audio_process_thread, pipeline._inference_thread):
        if t:
            assert not t.is_alive(), f"Thread {t.name} leaked!"


def test_mock_multimodal_adapter_direct():
    """Test MockMultimodalAdapter direct API."""
    adapter = MockMultimodalAdapter(simulated_latency=0.0)
    
    # Create mock frame
    t = torch.zeros((10, 10, 3), dtype=torch.uint8)
    class MockCap:
        def __init__(self):
            self.shape = (10, 10, 3)
            self.timestamp = 42.5
            self.__dlpack__ = t.__dlpack__
            self.__dlpack_device__ = t.__dlpack_device__
    frame = velo.Frame(MockCap(), None)
    
    t1 = Transcript(text="Hello world", start_timestamp=42.0, end_timestamp=43.0, confidence=0.98)
    resp = adapter.infer(
        frames=frame,
        transcripts=[t1],
        timestamp=42.5,
        prompt="What was spoken?",
    )

    assert resp.timestamp == 42.5
    assert len(resp.transcripts) == 1
    assert "Hello world" in resp.text
    assert "What was spoken?" in resp.text
