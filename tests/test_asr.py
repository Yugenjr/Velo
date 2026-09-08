import pytest
import torch
from velo.audio import AudioChunk
from velo.asr import MockASRAdapter, Transcript

def test_mock_asr():
    adapter = MockASRAdapter(mock_text="hello world", confidence=0.9)
    
    chunk = AudioChunk(
        samples=torch.zeros(8000, dtype=torch.float32),
        sample_rate=16000,
        channels=1,
        timestamp=10.0,
        duration=0.5
    )
    
    transcript = adapter.transcribe(chunk)
    assert isinstance(transcript, Transcript)
    assert transcript.text == "hello world"
    assert transcript.confidence == 0.9
    assert transcript.start_timestamp == 10.0
    assert transcript.end_timestamp == 10.5
