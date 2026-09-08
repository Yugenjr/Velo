"""
Multimodal LiveKit Ingestion & Temporal Fusion Test

Simulates simultaneous synchronized video and audio stream reception,
routing video to AIScheduler and audio to AudioScheduler, then verifying
multimodal alignment in TemporalFusion.
"""
import pytest
import torch
import time
import velo
from velo.audio import AudioChunk
from velo.audio_scheduler import AudioScheduler
from velo.fusion import TemporalFusion
from velo.asr import MockASRAdapter


class SynchronizedMediaStream:
    """Simulates a live Stream producing synchronized audio and video frames."""
    def __init__(self, fps=30, sample_rate=48000, channels=2, duration_s=1.0):
        self.fps = fps
        self.sample_rate = sample_rate
        self.channels = channels
        self.total_frames = int(fps * duration_s)
        self.total_audio_chunks = int(duration_s / 0.02) # 20ms chunks
        
        self.frame_idx = 0
        self.audio_idx = 0
        self.closed = False

    def next_frame(self):
        if self.closed or self.frame_idx >= self.total_frames:
            raise velo.StreamClosedError("End of video stream")
        ts = self.frame_idx / self.fps
        self.frame_idx += 1
        
        # Create a mock video capsule
        class MockCapsule:
            def __init__(self, timestamp):
                self.timestamp = timestamp
                self.shape = (1080, 1920, 3)
        return MockCapsule(ts), None

    def next_audio(self):
        if self.closed or self.audio_idx >= self.total_audio_chunks:
            raise velo.StreamClosedError("End of audio stream")
        ts = self.audio_idx * 0.02
        self.audio_idx += 1
        
        num_samples = int(0.02 * self.sample_rate * self.channels)
        pcm = torch.zeros(num_samples, dtype=torch.int16)
        raw_bytes = pcm.numpy().tobytes()
        return raw_bytes, ts, self.channels, self.sample_rate, 0.02

    def close(self):
        self.closed = True


def test_synchronized_multimodal_ingestion():
    raw_stream = SynchronizedMediaStream(fps=30, sample_rate=48000, channels=2, duration_s=1.0)
    stream = velo.Stream(raw_stream)
    
    audio_scheduler = AudioScheduler(chunk_duration_s=0.1, sample_rate=48000, channels=2)
    fusion = TemporalFusion(max_history_s=10.0)
    asr = MockASRAdapter(mock_text="multimodal synchronized query")
    
    # Ingest entire 1.0s video and audio stream
    for _ in range(raw_stream.total_frames):
        frame = stream.next()
        fusion.add_video(frame, timestamp=frame.timestamp)
        
    for _ in range(raw_stream.total_audio_chunks):
        chunk = stream.next_audio()
        audio_scheduler.submit(chunk)
        
    # Drain audio scheduler and add transcripts to fusion
    for _ in range(10):
        chunk = audio_scheduler.acquire()
        transcript = asr.transcribe(chunk)
        fusion.add_audio(transcript, timestamp=transcript.start_timestamp, duration=chunk.duration)
        
    # Verify temporal alignment
    # Query at T = 0.5s with window 0.15s (covers [0.35, 0.65])
    ctx = fusion.context(timestamp=0.5, window=0.15)
    
    # Video frames at 30fps: ~9-10 frames in [0.35, 0.65]
    assert len(ctx.video) >= 8
    # Audio chunks of 100ms in [0.35, 0.65]: chunks at 0.3, 0.4, 0.5, 0.6 overlap
    assert len(ctx.audio) >= 3
    
    # Verify latest context
    latest_ctx = fusion.latest_context(window=0.2)
    assert latest_ctx is not None
    assert latest_ctx.reference_timestamp >= 0.95
    assert len(latest_ctx.video) > 0
    assert len(latest_ctx.audio) > 0
