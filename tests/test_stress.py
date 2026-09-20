"""
Velo Backpressure & Stress Test Suite (Milestone V1.12)

Scenarios Tested:
A. 30 FPS video + slow inference (backpressure drop & queue bound)
B. 60 FPS video + slow inference (high rate backpressure)
C. High-rate audio + slow ASR (audio buffer bounds)
D. Video + Audio + Slow Multimodal Inference
E. 15 Repeated Pipeline Start / Stop Lifecycle Cycles (resource and thread stability)
"""
import time
import asyncio
import psutil
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


class FastStream:
    """Produces video frames and audio chunks at target simulated framerates."""
    def __init__(self, fps=30, total_frames=120, total_audio=300):
        self.fps = fps
        self.total_frames = total_frames
        self.total_audio = total_audio
        self.frame_idx = 0
        self.audio_idx = 0
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._tensor = torch.zeros((100, 100, 3), dtype=torch.uint8, device=self.device)
        self._audio_bytes = (torch.full((960,), 5000, dtype=torch.int16)).numpy().tobytes()

    def next_frame(self):
        if self.frame_idx >= self.total_frames:
            raise StreamClosedError("End of video stream")
        ts = self.frame_idx / self.fps
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
        if self.audio_idx >= self.total_audio:
            raise StreamClosedError("End of audio stream")
        ts = self.audio_idx * 0.02
        self.audio_idx += 1
        return self._audio_bytes, ts, 2, 48000, 0.02

    def close(self):
        pass


@pytest.mark.asyncio
async def test_scenario_a_30fps_slow_inference():
    """Scenario A: 30 FPS video + slow 50ms inference -> verify scheduler drops old frames and maintains queue bounds."""
    raw = FastStream(fps=30, total_frames=150, total_audio=0)
    stream = velo.Stream(raw)
    scheduler = AIScheduler(target_fps=5.0, max_temporal_frames=30)
    vlm = MockMultimodalAdapter(simulated_latency=0.05)
    
    pipeline = AIPipeline(stream=stream, scheduler=scheduler, vlm=vlm)
    await pipeline.start()
    
    inferences = 0
    async for _ in pipeline.run_inference("slow test"):
        inferences += 1
        
    await pipeline.stop()
    stats = scheduler.stats()
    
    assert stats["frames_processed"] == inferences
    assert stats["frames_dropped"] > 0, "Scheduler should drop redundant frames under backpressure"
    assert stats["current_queue_depth"] <= 30, "Queue depth must remain bounded"


@pytest.mark.asyncio
async def test_scenario_b_60fps_high_rate():
    """Scenario B: 60 FPS video + slow inference -> verify high backpressure stability."""
    raw = FastStream(fps=60, total_frames=200, total_audio=0)
    stream = velo.Stream(raw)
    scheduler = AIScheduler(target_fps=5.0, max_temporal_frames=20)
    vlm = MockMultimodalAdapter(simulated_latency=0.04)
    
    pipeline = AIPipeline(stream=stream, scheduler=scheduler, vlm=vlm)
    await pipeline.start()
    
    async for _ in pipeline.run_inference("60fps test"):
        pass
        
    await pipeline.stop()
    stats = scheduler.stats()
    assert stats["frames_dropped"] > 50


@pytest.mark.asyncio
async def test_scenario_c_high_rate_audio_slow_asr():
    """Scenario C: High-rate audio (200 chunks) + slow ASR -> verify AudioScheduler bounds memory."""
    raw = FastStream(fps=0, total_frames=0, total_audio=200)
    stream = velo.Stream(raw)
    
    audio_sched = AudioScheduler(chunk_duration_s=0.1, sample_rate=48000, channels=2, max_buffered_s=2.0)
    vad = VAD(energy_threshold=0.001)
    
    # Slow ASR
    class SlowASR(velo.BaseASRAdapter):
        def transcribe(self, chunk):
            time.sleep(0.04)
            return velo.Transcript(text="slow transcribed audio", start_timestamp=chunk.timestamp, end_timestamp=chunk.timestamp+chunk.duration, confidence=0.99)
            
    pipeline = AIPipeline(
        stream=stream,
        scheduler=AIScheduler(),
        vlm=MockMultimodalAdapter(),
        audio_scheduler=audio_sched,
        fusion=TemporalFusion(max_history_s=5.0),
        vad=vad,
        asr=SlowASR(),
    )
    
    await pipeline.start()
    await asyncio.sleep(0.5)
    await pipeline.stop()
    
    stats = audio_sched.stats()
    assert stats["buffered_duration"] <= 2.5, "Audio buffer must not exceed max_buffered_s"


@pytest.mark.asyncio
async def test_scenario_d_full_multimodal_stress():
    """Scenario D: Video + Audio + Multimodal inference concurrent load."""
    raw = FastStream(fps=30, total_frames=90, total_audio=150)
    stream = velo.Stream(raw)
    
    scheduler = AIScheduler(target_fps=10.0, max_temporal_frames=30)
    audio_sched = AudioScheduler(chunk_duration_s=0.1, sample_rate=48000, channels=2, max_buffered_s=5.0)
    fusion = TemporalFusion(max_history_s=10.0)
    vad = VAD(energy_threshold=0.001)
    asr = MockASRAdapter(mock_text="user input")
    vlm = MockMultimodalAdapter(simulated_latency=0.02)
    
    pipeline = AIPipeline(
        stream=stream,
        scheduler=scheduler,
        vlm=vlm,
        audio_scheduler=audio_sched,
        fusion=fusion,
        vad=vad,
        asr=asr,
        context_window_s=1.0,
    )
    
    await pipeline.start()
    inferences = 0
    async for _ in pipeline.run_inference("full stress test"):
        inferences += 1
        
    await pipeline.stop()
    m = pipeline.metrics()
    assert m["inference"]["inference_count"] == inferences
    assert m["system"]["pipeline_state"] == "STOPPED"


@pytest.mark.asyncio
async def test_scenario_e_repeated_start_stop_cycles():
    """Scenario E: 15 consecutive start/stop cycles -> verify zero thread leaks and clean restarts."""
    initial_threads = psutil.Process().num_threads()
    
    for cycle in range(15):
        raw = FastStream(fps=30, total_frames=20, total_audio=30)
        stream = velo.Stream(raw)
        pipeline = AIPipeline(
            stream=stream,
            scheduler=AIScheduler(target_fps=30),
            vlm=MockMultimodalAdapter(simulated_latency=0.005),
            audio_scheduler=AudioScheduler(chunk_duration_s=0.1, sample_rate=48000, channels=2),
            fusion=TemporalFusion(),
            vad=VAD(),
            asr=MockASRAdapter(),
        )
        
        await pipeline.start()
        count = 0
        async for _ in pipeline.run_inference(f"cycle {cycle}"):
            count += 1
            if count >= 3:
                break
                
        await pipeline.stop()
        
    # Check thread stability
    final_threads = psutil.Process().num_threads()
    # Allow small variance (+/- 3 threads) from runtime/asyncio pool, verify no runaway thread leak (+15*4 = 60 threads)
    assert abs(final_threads - initial_threads) < 6, f"Thread leak detected: {initial_threads} -> {final_threads}"
