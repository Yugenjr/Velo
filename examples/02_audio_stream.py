"""
Example 02: Audio Stream Ingestion, Chunking & VAD

Demonstrates ingesting Opus audio chunks, buffering in AudioScheduler,
and detecting active speech segments via Voice Activity Detection (VAD).
"""
import velo
import torch

def main():
    # 1. Configure audio scheduler (500ms slices from 20ms Opus frames)
    audio_scheduler = velo.AudioScheduler(
        chunk_duration_s=0.5,
        sample_rate=48000,
        channels=2,
        max_buffered_s=10.0,
    )
    
    # 2. Configure VAD with energy threshold
    vad = velo.VAD(energy_threshold=0.01)
    
    # 3. Simulate an AudioChunk (float32 tensor normalized in [-1.0, 1.0])
    samples = torch.sin(torch.linspace(0, 100, 48000 * 2)) * 0.5
    chunk = velo.AudioChunk(
        samples=samples,
        sample_rate=48000,
        channels=2,
        timestamp=0.0,
        duration=1.0,
    )
    
    # 4. Evaluate VAD
    res = vad.analyze(chunk)
    print(f"VAD Analysis: is_speech={res.is_speech}, energy={res.energy:.4f}")

if __name__ == "__main__":
    main()
