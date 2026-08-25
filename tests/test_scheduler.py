import time
import threading
import pytest
from velo import AIScheduler, SchedulerClosedError

class MockFrame:
    def __init__(self, index):
        self.index = index

def test_scheduler_enforces_target_fps():
    scheduler = AIScheduler(target_fps=10.0, max_queue=5)
    # 10 FPS = 0.1s per frame
    
    # Submit 3 frames instantly
    for i in range(3):
        scheduler.submit(MockFrame(i))
        
    t0 = time.time()
    f1 = scheduler.next()
    f2 = scheduler.next()
    f3 = scheduler.next()
    t1 = time.time()
    
    # Getting 3 frames at 10 FPS means 2 intervals between the 3 frames
    # The first frame returns immediately. The second at +0.1s, third at +0.2s.
    assert t1 - t0 >= 0.15, f"Elapsed time {t1 - t0} is too fast for 10 FPS"
    assert f1.index == 0
    assert f2.index == 1
    assert f3.index == 2

def test_scheduler_queue_overflow_latest_frame():
    scheduler = AIScheduler(target_fps=100.0, max_queue=2)
    
    # Submit 5 frames instantly
    for i in range(5):
        scheduler.submit(MockFrame(i))
        
    # Queue size is 2, strategy="latest" drops oldest. 
    # State after submissions: [3, 4]
    
    f1 = scheduler.next()
    f2 = scheduler.next()
    
    assert f1.index == 3
    assert f2.index == 4
    
    stats = scheduler.stats()
    assert stats["frames_received"] == 5
    assert stats["frames_dropped"] == 3
    assert stats["frames_processed"] == 2

def test_scheduler_concurrent_producer_consumer():
    scheduler = AIScheduler(target_fps=20.0, max_queue=10)
    consumed = []
    
    def consumer():
        try:
            while True:
                consumed.append(scheduler.next().index)
        except SchedulerClosedError:
            pass
            
    def producer():
        for i in range(15):
            scheduler.submit(MockFrame(i))
            time.sleep(0.01) # Produce at 100 FPS
        scheduler.close()
        
    t1 = threading.Thread(target=consumer)
    t2 = threading.Thread(target=producer)
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
    assert len(consumed) > 0
    assert consumed[-1] == 14
    stats = scheduler.stats()
    assert stats["frames_received"] == 15

def test_scheduler_close_behavior():
    scheduler = AIScheduler(target_fps=50.0, max_queue=5)
    
    scheduler.submit(MockFrame(0))
    scheduler.submit(MockFrame(1))
    scheduler.close()
    
    # Should still yield remaining frames before raising
    f1 = scheduler.next()
    assert f1.index == 0
    f2 = scheduler.next()
    assert f2.index == 1
    
    with pytest.raises(SchedulerClosedError):
        scheduler.next()

def test_scheduler_stats():
    scheduler = AIScheduler(target_fps=100.0, max_queue=5)
    
    for i in range(3):
        scheduler.submit(MockFrame(i))
        
    scheduler.next()
    scheduler.next()
    
    stats = scheduler.stats()
    assert stats["frames_received"] == 3
    assert stats["frames_processed"] == 2
    assert stats["frames_dropped"] == 0
    assert stats["average_queue_depth"] > 0
    assert stats["average_frame_age"] >= 0
