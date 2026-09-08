import pytest
import torch
import time
import threading
from velo.audio import AudioChunk
from velo.audio_scheduler import AudioScheduler, SchedulerClosedError

def test_audio_scheduler_chunking():
    scheduler = AudioScheduler(chunk_duration_s=0.5, max_buffered_s=2.0, sample_rate=16000)
    
    # Submit 0.1s chunks
    for i in range(5):
        chunk = AudioChunk(
            samples=torch.zeros(1600, dtype=torch.float32),
            sample_rate=16000,
            channels=1,
            timestamp=i * 0.1,
            duration=0.1
        )
        scheduler.submit(chunk)
        
    # We should be able to acquire exactly one 0.5s chunk
    out_chunk = scheduler.acquire()
    assert out_chunk.duration == 0.5
    assert out_chunk.samples.size(0) == 8000
    assert out_chunk.timestamp == 0.0 # First chunk's timestamp
    
    stats = scheduler.stats()
    assert stats["chunks_received"] == 5
    assert stats["chunks_processed"] == 1
    assert stats["chunks_dropped"] == 0

def test_audio_scheduler_backpressure():
    scheduler = AudioScheduler(chunk_duration_s=0.5, max_buffered_s=1.0, sample_rate=16000)
    
    # Submit 3 seconds of audio (should drop 2 seconds)
    for i in range(30):
        chunk = AudioChunk(
            samples=torch.zeros(1600, dtype=torch.float32),
            sample_rate=16000,
            channels=1,
            timestamp=i * 0.1,
            duration=0.1
        )
        scheduler.submit(chunk)
    
    stats = scheduler.stats()
    # It drops from the front of the queue
    assert stats["chunks_received"] == 30
    assert stats["chunks_dropped"] > 0
    assert stats["buffered_duration"] <= 1.0
    
def test_audio_scheduler_close():
    scheduler = AudioScheduler()
    
    def acquire_thread():
        with pytest.raises(SchedulerClosedError):
            scheduler.acquire()
            
    t = threading.Thread(target=acquire_thread)
    t.start()
    
    time.sleep(0.1)
    scheduler.close()
    t.join(timeout=1.0)
    assert not t.is_alive()
