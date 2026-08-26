import pytest
import time
import torch
import threading
from velo import AIScheduler
from velo.scene import SceneChangeDetector

class MockFrame:
    def __init__(self, shape=(1080, 1920, 4), val=0):
        self.shape = shape
        self.timestamp = time.time()
        # Ensure we have a float-castable or valid uint8 tensor representing an image
        self._tensor = torch.full(shape, val, dtype=torch.uint8)
        if torch.cuda.is_available():
            self._tensor = self._tensor.to("cuda")

    def to_torch(self):
        return self._tensor

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU scene tests")
def test_scene_first_frame():
    detector = SceneChangeDetector(threshold=0.05)
    frame = MockFrame(val=10)
    result = detector.update(frame)
    
    assert result.changed is True
    assert result.score == 1.0 # First frame should force a 1.0 score and True
    assert detector.reference_tensor is not None
    assert detector.reference_tensor.device.type == "cuda"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU scene tests")
def test_scene_identical_frames():
    detector = SceneChangeDetector(threshold=0.05)
    frame = MockFrame(val=50)
    
    res1 = detector.update(frame)
    assert res1.changed is True
    
    res2 = detector.update(frame)
    assert res2.changed is False
    assert res2.score == 0.0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU scene tests")
def test_scene_small_visual_change():
    detector = SceneChangeDetector(threshold=0.05)
    # val represents pixel intensity 0-255. 0.05 threshold is roughly 12.7 / 255.
    f1 = MockFrame(val=100)
    detector.update(f1)
    
    # Small change (5/255 = 0.019 < 0.05)
    f2 = MockFrame(val=105)
    res2 = detector.update(f2)
    
    assert res2.changed is False
    assert res2.score < 0.05

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU scene tests")
def test_scene_large_visual_change():
    detector = SceneChangeDetector(threshold=0.05)
    f1 = MockFrame(val=100)
    detector.update(f1)
    
    # Large change (150/255 = 0.58 > 0.05)
    f2 = MockFrame(val=250)
    res2 = detector.update(f2)
    
    assert res2.changed is True
    assert res2.score > 0.05

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU scene tests")
def test_scheduler_integration():
    scheduler = AIScheduler(target_fps=1000.0, scene_aware=True, scene_threshold=0.05)
    
    # First frame to initialize baseline
    scheduler.submit(MockFrame(val=100))
    frame1 = scheduler.acquire()
    scheduler.release()
    
    # We will run acquire() in a background thread so it can block
    # while we submit identical frames, then a changed frame.
    result_frame = []
    
    def inference_worker():
        f = scheduler.acquire()
        result_frame.append(f)
        scheduler.release()
        
    t = threading.Thread(target=inference_worker)
    t.start()
    
    # Yield to let acquire() block
    time.sleep(0.1)
    
    # Submit identical frames (these should be skipped)
    scheduler.submit(MockFrame(val=100))
    scheduler.submit(MockFrame(val=100))
    scheduler.submit(MockFrame(val=100))
    
    # Yield to let acquire() process and skip them
    time.sleep(0.2)
    
    # Submit changed frame (this should wake it up and be acquired)
    scheduler.submit(MockFrame(val=250))
    
    t.join(timeout=2.0)
    
    stats = scheduler.stats()
    
    # It should have skipped the 3 identical frames we submitted while it was waiting
    assert stats["frames_skipped_scene"] >= 1
    
    # Snapshots should still contain all submitted frames!
    snaps = scheduler.snapshot()
    assert len(snaps) >= 4

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU scene tests")
def test_scheduler_disabled():
    scheduler = AIScheduler(target_fps=1000.0, scene_aware=False)
    scheduler.submit(MockFrame(val=100))
    frame = scheduler.acquire()
    assert frame is not None
    scheduler.release()
    
    scheduler.submit(MockFrame(val=100)) # identical
    frame2 = scheduler.acquire()
    assert frame2 is not None # It returned it anyway
    scheduler.release()
    
    stats = scheduler.stats()
    assert stats.get("frames_skipped_scene", 0) == 0
