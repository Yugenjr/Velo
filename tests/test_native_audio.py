import pytest
import torch
import time
import velo
from velo.audio import AudioChunk
from velo.audio_scheduler import AudioScheduler
from velo.vad import VAD
from velo.asr import MockASRAdapter
from velo.fusion import TemporalFusion
from velo.exceptions import StreamClosedError


class MockNativeAudioStream:
    """Mock simulating native Rust WebRTC stream with audio frames."""
    def __init__(self, sample_rate=48000, channels=2, num_chunks=10):
        self.sample_rate = sample_rate
        self.channels = channels
        self.num_chunks = num_chunks
        self.current_chunk = 0
        self.closed = False
        self.dropped_frames = 0
        self.decode_errors = 0

    def next_frame(self):
        raise StreamClosedError("No video in audio-only mock")

    def next_audio(self):
        if self.closed or self.current_chunk >= self.num_chunks:
            raise StreamClosedError("Stream closed")
        
        # 20ms chunk at 48kHz stereo = 960 samples/ch = 1920 total samples
        samples_per_ch = int(0.02 * self.sample_rate)
        total_samples = samples_per_ch * self.channels
        
        # Generate int16 PCM bytes
        t = torch.linspace(0, 0.02, total_samples)
        pcm_tensor = (32767.0 * 0.5 * torch.sin(2 * 3.14159 * 440 * t)).to(torch.int16)
        raw_bytes = pcm_tensor.numpy().tobytes()
        
        timestamp = self.current_chunk * 0.02
        duration = 0.02
        self.current_chunk += 1
        return raw_bytes, timestamp, self.channels, self.sample_rate, duration

    def close(self):
        self.closed = True


def test_stream_next_audio():
    mock_native = MockNativeAudioStream(sample_rate=48000, channels=2, num_chunks=5)
    stream = velo.Stream(mock_native)
    
    chunk = stream.next_audio()
    assert isinstance(chunk, AudioChunk)
    assert chunk.sample_rate == 48000
    assert chunk.channels == 2
    assert chunk.timestamp == 0.0
    assert abs(chunk.duration - 0.02) < 1e-4
    assert isinstance(chunk.samples, torch.Tensor)
    assert chunk.samples.dtype == torch.float32
    assert chunk.samples.numel() == 960 * 2
    # Verify values are in normalized range [-1.0, 1.0]
    assert chunk.samples.min() >= -1.0
    assert chunk.samples.max() <= 1.0


def test_stream_audio_closure():
    mock_native = MockNativeAudioStream(sample_rate=48000, channels=2, num_chunks=2)
    stream = velo.Stream(mock_native)
    
    _ = stream.next_audio()
    _ = stream.next_audio()
    
    with pytest.raises(StreamClosedError):
        stream.next_audio()


def test_native_audio_into_audio_scheduler():
    mock_native = MockNativeAudioStream(sample_rate=48000, channels=2, num_chunks=30)
    stream = velo.Stream(mock_native)
    
    # AudioScheduler yielding 0.1s chunks (5 x 20ms chunks per yield)
    scheduler = AudioScheduler(chunk_duration_s=0.1, sample_rate=48000, channels=2)
    
    # Ingest 10 chunks (0.2s of audio)
    for _ in range(10):
        chunk = stream.next_audio()
        scheduler.submit(chunk)
        
    chunk_out1 = scheduler.acquire()
    assert chunk_out1.sample_rate == 48000
    assert chunk_out1.channels == 2
    assert chunk_out1.timestamp == 0.0
    assert abs(chunk_out1.duration - 0.1) < 1e-4
    assert chunk_out1.samples.numel() == int(0.1 * 48000 * 2)
    
    chunk_out2 = scheduler.acquire()
    assert abs(chunk_out2.timestamp - 0.1) < 1e-4


def test_native_audio_multimodal_fusion():
    fusion = TemporalFusion(max_history_s=10.0)
    vad = VAD(energy_threshold=0.01)
    asr = MockASRAdapter(mock_text="WebRTC live voice")
    
    mock_native = MockNativeAudioStream(sample_rate=48000, channels=2, num_chunks=50)
    stream = velo.Stream(mock_native)
    scheduler = AudioScheduler(chunk_duration_s=0.2, sample_rate=48000, channels=2)
    
    # Submit 1.0s of audio
    for _ in range(50):
        scheduler.submit(stream.next_audio())
        
    # Process audio through VAD and ASR, add to fusion
    for _ in range(5):
        chunk = scheduler.acquire()
        vad_res = vad.analyze(chunk)
        if vad_res.is_speech:
            transcript = asr.transcribe(chunk)
            fusion.add_audio(transcript, timestamp=transcript.start_timestamp, duration=chunk.duration)
            
    # Add dummy video frames
    class MockFrame:
        def __init__(self, ts):
            self.timestamp = ts
            
    fusion.add_video(MockFrame(0.1), timestamp=0.1)
    fusion.add_video(MockFrame(0.5), timestamp=0.5)
    fusion.add_video(MockFrame(0.9), timestamp=0.9)
    
    # Query context around 0.5s
    ctx = fusion.context(timestamp=0.5, window=0.25)
    assert len(ctx.video) == 1
    assert ctx.video[0].timestamp == 0.5
    # Audio chunks spanning [0.2, 0.4], [0.4, 0.6], [0.6, 0.8] overlap with [0.25, 0.75]
    assert len(ctx.audio) >= 2
    assert ctx.audio[0].payload.text == "WebRTC live voice"
