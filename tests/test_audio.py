import pytest
import torch
from velo.audio import AudioChunk

def test_audio_chunk_initialization():
    samples = torch.zeros(16000, dtype=torch.float32)
    chunk = AudioChunk(
        samples=samples,
        sample_rate=16000,
        channels=1,
        timestamp=1.5,
        duration=1.0
    )
    
    assert chunk.sample_rate == 16000
    assert chunk.channels == 1
    assert chunk.timestamp == 1.5
    assert chunk.duration == 1.0
    assert torch.equal(chunk.samples, samples)

def test_audio_chunk_validation():
    with pytest.raises(TypeError):
        AudioChunk(
            samples=[0, 0, 0], # not a tensor
            sample_rate=16000,
            channels=1,
            timestamp=0.0,
            duration=1.0
        )
        
    with pytest.raises(TypeError):
        AudioChunk(
            samples=torch.zeros(16),
            sample_rate="16000", # not an int
            channels=1,
            timestamp=0.0,
            duration=1.0
        )
