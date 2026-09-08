import pytest
import time
import threading
from velo.fusion import TemporalFusion
from velo.core import Frame

class MockFrame:
    """Mock for velo.core.Frame"""
    def __init__(self, timestamp=0.0):
        self._ts = timestamp
        
    @property
    def timestamp(self):
        return self._ts
        
class MockTranscript:
    """Mock for velo.asr.Transcript"""
    def __init__(self, text=""):
        self.text = text


def test_fusion_ordering_and_retrieval():
    fusion = TemporalFusion(max_history_s=10.0)
    
    # Add video frames out of order
    fusion.add_video(MockFrame(1.5), timestamp=1.5)
    fusion.add_video(MockFrame(1.1), timestamp=1.1)
    fusion.add_video(MockFrame(1.3), timestamp=1.3)
    
    assert len(fusion._video_buffer) == 3
    assert fusion._video_buffer[0].timestamp == 1.1
    assert fusion._video_buffer[1].timestamp == 1.3
    assert fusion._video_buffer[2].timestamp == 1.5

def test_fusion_windowing():
    fusion = TemporalFusion(max_history_s=10.0)
    
    # Video at 10.0, 11.0, 12.0
    fusion.add_video(MockFrame(10.0), timestamp=10.0)
    fusion.add_video(MockFrame(11.0), timestamp=11.0)
    fusion.add_video(MockFrame(12.0), timestamp=12.0)
    
    # Audio spanning 10.5 to 11.5
    fusion.add_audio(MockTranscript("hello"), timestamp=10.5, duration=1.0)
    # Audio spanning 11.8 to 12.2
    fusion.add_audio(MockTranscript("world"), timestamp=11.8, duration=0.4)
    
    # Query at 11.0 with window 0.6 (10.4 to 11.6)
    ctx = fusion.context(timestamp=11.0, window=0.6)
    
    assert len(ctx.video) == 1
    assert ctx.video[0].timestamp == 11.0
    
    assert len(ctx.audio) == 1
    assert ctx.audio[0].payload.text == "hello"

def test_fusion_memory_bounds():
    fusion = TemporalFusion(max_history_s=2.0)
    
    # Add items over a 5 second span
    fusion.add_video(MockFrame(1.0), timestamp=1.0)
    fusion.add_video(MockFrame(2.0), timestamp=2.0)
    fusion.add_video(MockFrame(3.0), timestamp=3.0)
    fusion.add_video(MockFrame(4.0), timestamp=4.0)
    fusion.add_video(MockFrame(5.0), timestamp=5.0)
    
    # Since max history is 2.0, adding frame at 5.0 should evict anything before 3.0
    assert len(fusion._video_buffer) == 3
    assert fusion._video_buffer[0].timestamp == 3.0
    assert fusion._video_buffer[-1].timestamp == 5.0

def test_fusion_latest_context():
    fusion = TemporalFusion(max_history_s=10.0)
    
    fusion.add_video(MockFrame(10.0), timestamp=10.0)
    fusion.add_audio(MockTranscript("audio"), timestamp=10.5, duration=1.0) # Ends at 11.5
    
    ctx = fusion.latest_context(window=2.0)
    assert ctx is not None
    assert ctx.reference_timestamp == 11.5 # The max timestamp seen
    
    assert len(ctx.video) == 1
    assert len(ctx.audio) == 1

def test_fusion_thread_safety():
    fusion = TemporalFusion(max_history_s=10.0)
    
    def add_video_worker():
        for i in range(100):
            fusion.add_video(MockFrame(float(i)), timestamp=float(i))
            time.sleep(0.001)
            
    def add_audio_worker():
        for i in range(100):
            fusion.add_audio(MockTranscript(str(i)), timestamp=float(i), duration=0.5)
            time.sleep(0.001)
            
    t1 = threading.Thread(target=add_video_worker)
    t2 = threading.Thread(target=add_audio_worker)
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
    # Just checking it doesn't crash and evicts correctly
    # The latest timestamp should be 99.something, meaning we only keep the last 10 seconds (roughly 90.0 to 99.0)
    assert len(fusion._video_buffer) <= 12
    assert len(fusion._audio_buffer) <= 12
