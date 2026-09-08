import pytest
import torch
from velo.audio import AudioChunk
from velo.audio_scheduler import AudioScheduler
from velo.vad import VAD
from velo.asr import MockASRAdapter

def test_audio_pipeline_end_to_end():
    # 1. Setup
    scheduler = AudioScheduler(chunk_duration_s=0.5, sample_rate=16000)
    vad = VAD(energy_threshold=0.01)
    asr = MockASRAdapter(mock_text="I am speaking now")
    
    # 2. Ingest fake audio (1 second total, in two 0.5s chunks)
    # First chunk is silence
    scheduler.submit(AudioChunk(
        samples=torch.zeros(8000, dtype=torch.float32),
        sample_rate=16000,
        channels=1,
        timestamp=0.0,
        duration=0.5
    ))
    
    # Second chunk is speech
    t = torch.linspace(0, 1.0, 8000)
    speech_samples = 0.5 * torch.sin(2 * 3.14159 * 440 * t)
    scheduler.submit(AudioChunk(
        samples=speech_samples.to(torch.float32),
        sample_rate=16000,
        channels=1,
        timestamp=0.5,
        duration=0.5
    ))
    
    # 3. Process first chunk (silence)
    chunk1 = scheduler.acquire()
    vad_res1 = vad.analyze(chunk1)
    assert vad_res1.is_speech is False
    # In a real pipeline, we would skip ASR here.
    
    # 4. Process second chunk (speech)
    chunk2 = scheduler.acquire()
    vad_res2 = vad.analyze(chunk2)
    assert vad_res2.is_speech is True
    
    # 5. Run ASR on speech chunk
    transcript = asr.transcribe(chunk2)
    assert transcript.text == "I am speaking now"
    assert transcript.start_timestamp == 0.5
    assert transcript.end_timestamp == 1.0
