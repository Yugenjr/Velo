import pytest
import torch
import time
from velo.scheduler import AIScheduler
from velo.core import Frame

class MockFrame:
    def __init__(self, tensor):
        self._tensor = tensor
    def to_torch(self):
        return self._tensor

def test_candidate_aware_scheduler():
    scheduler = AIScheduler(
        target_fps=10.0,
        candidate_aware=True,
        candidate_cooldown_ms=0.0,
        candidate_threshold=0.1
    )
    
    # Send identical frames
    tensor_base = torch.ones((480, 640, 4), dtype=torch.uint8) * 128
    tensor_diff = torch.zeros((480, 640, 4), dtype=torch.uint8)
    
    frame1 = MockFrame(tensor_base)
    frame2 = MockFrame(tensor_base)
    frame3 = MockFrame(tensor_diff)
    
    # 1. First frame (novelty)
    scheduler.submit(frame1)
    acq1 = scheduler.acquire()
    assert acq1 is frame1
    scheduler.release()
    
    # 2. Identical frame (should be rejected)
    scheduler.submit(frame2)
    
    import threading
    def submit_delayed():
        time.sleep(0.1)
        scheduler.submit(frame3)
        
    threading.Thread(target=submit_delayed, daemon=True).start()
    
    acq2 = scheduler.acquire()
    assert acq2 is frame3 # We skipped frame2!
    scheduler.release()
    
    stats = scheduler.stats()
    assert stats["candidates_evaluated"] == 3
    assert stats["candidates_accepted"] == 2
    assert stats["candidates_rejected"] == 1
    assert stats["candidate_acceptance_rate"] == (2.0 / 3.0)
