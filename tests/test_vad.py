import pytest
import torch
from velo.audio import AudioChunk
from velo.vad import VAD

def test_vad_silence():
    vad = VAD(energy_threshold=0.01)
    
    # Absolute silence
    chunk = AudioChunk(
        samples=torch.zeros(8000, dtype=torch.float32),
        sample_rate=16000,
        channels=1,
        timestamp=0.0,
        duration=0.5
    )
    
    result = vad.analyze(chunk)
    assert result.is_speech is False
    assert result.energy == 0.0

def test_vad_speech():
    vad = VAD(energy_threshold=0.01)
    
    # Loud noise
    # Sinewave with amplitude 0.5 -> RMS = ~0.35, which is > 0.01
    t = torch.linspace(0, 1.0, 8000)
    samples = 0.5 * torch.sin(2 * 3.14159 * 440 * t)
    
    chunk = AudioChunk(
        samples=samples.to(torch.float32),
        sample_rate=16000,
        channels=1,
        timestamp=0.0,
        duration=0.5
    )
    
    result = vad.analyze(chunk)
    assert result.is_speech is True
    assert result.energy > 0.01

def test_vad_empty():
    vad = VAD(energy_threshold=0.01)
    chunk = AudioChunk(
        samples=torch.tensor([], dtype=torch.float32),
        sample_rate=16000,
        channels=1,
        timestamp=0.0,
        duration=0.0
    )
    result = vad.analyze(chunk)
    assert result.is_speech is False
    assert result.energy == 0.0
