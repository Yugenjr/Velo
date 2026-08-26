import time
import threading
import pytest
from velo import AIScheduler, SchedulerClosedError

class MockFrame:
    def __init__(self, index, timestamp=None):
        self.index = index
        if timestamp is not None:
            self.timestamp = timestamp

def test_scheduler_acquire_release_fps():
    scheduler = AIScheduler(target_fps=10.0, temporal_window=5.0, max_temporal_frames=10)
    
    # Producer thread simulating 100 FPS
    def producer():
        for i in range(10):
            scheduler.submit(MockFrame(i))
            time.sleep(0.01)
            
    t1 = threading.Thread(target=producer)
    t1.start()
    
    t0 = time.time()
    
    # 10 FPS = 0.1s per frame
    f1 = scheduler.acquire()
    scheduler.release()
    
    f2 = scheduler.acquire()
    scheduler.release()
    
    t1_end = time.time()
    
    # Time for 2 frames at 10 FPS is ~0.1s
    assert t1_end - t0 >= 0.08
    t1.join()
    scheduler.close()

def test_scheduler_temporal_buffer_snapshot():
    scheduler = AIScheduler(target_fps=100.0, temporal_window=1.0, max_temporal_frames=10)
    
    # Submit 5 frames across 0.5s
    for i in range(5):
        scheduler.submit(MockFrame(i, timestamp=time.time()))
        time.sleep(0.1)
        
    frames = scheduler.snapshot(duration=2.0)
    assert len(frames) == 5
    assert [f.index for f in frames] == [0, 1, 2, 3, 4]
    
    # Test shorter duration
    frames_short = scheduler.snapshot(duration=0.25)
    # The frames are spaced by 0.1s, so 0.25s should grab the last 2 or 3
    assert len(frames_short) in (2, 3)
    assert frames_short[-1].index == 4

def test_scheduler_temporal_eviction_max_frames():
    scheduler = AIScheduler(target_fps=100.0, temporal_window=5.0, max_temporal_frames=3)
    
    for i in range(5):
        scheduler.submit(MockFrame(i))
        
    frames = scheduler.snapshot(duration=5.0)
    # The max_frames is 3, so only 2, 3, 4 should be retained
    assert len(frames) == 3
    assert frames[0].index == 2
    assert frames[2].index == 4
    
    stats = scheduler.stats()
    assert stats["frames_dropped_stale"] == 2

def test_scheduler_backpressure():
    scheduler = AIScheduler(target_fps=20.0, temporal_window=5.0, max_temporal_frames=100)
    
    f1 = MockFrame(1)
    f2 = MockFrame(2)
    f3 = MockFrame(3)
    
    # Submit first frame
    scheduler.submit(f1)
    
    # Acquire it (worker gets busy)
    acq1 = scheduler.acquire()
    assert acq1.index == 1
    
    # Submit more while busy
    scheduler.submit(f2)
    scheduler.submit(f3)
    
    # Release the first frame
    scheduler.release()
    
    # Acquire next. We expect to get f3 because it's the latest, but wait!
    # f2 should be marked as backpressure dropped.
    acq2 = scheduler.acquire()
    assert acq2.index == 3
    scheduler.release()
    
    stats = scheduler.stats()
    assert stats["frames_dropped_backpressure"] >= 1
    assert stats["inference_busy_time"] > 0

def test_scheduler_close_behavior():
    scheduler = AIScheduler(target_fps=50.0)
    scheduler.submit(MockFrame(0))
    scheduler.close()
    
    f1 = scheduler.acquire()
    scheduler.release()
    assert f1.index == 0
    
    with pytest.raises(SchedulerClosedError):
        scheduler.acquire()

def test_scheduler_stats():
    scheduler = AIScheduler(target_fps=100.0, temporal_window=5.0)
    
    scheduler.submit(MockFrame(0, timestamp=time.time() - 0.5))
    scheduler.submit(MockFrame(1, timestamp=time.time() - 0.1))
    
    scheduler.acquire()
    scheduler.release()
    
    scheduler.submit(MockFrame(2, timestamp=time.time()))
    scheduler.acquire()
    scheduler.release()
    
    stats = scheduler.stats()
    assert stats["frames_received"] == 3
    assert stats["frames_processed"] == 2
    assert stats["average_frame_age"] > 0
    assert stats["effective_inference_fps"] > 0
