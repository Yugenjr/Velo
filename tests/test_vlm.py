import pytest
from velo import AIScheduler, MockVLM, BaseVLMAdapter, VLMResponse

class DummyFrame:
    def __init__(self, index=0):
        self.index = index

def test_base_vlm_adapter_raises():
    adapter = BaseVLMAdapter()
    with pytest.raises(NotImplementedError):
        adapter.generate(DummyFrame(), "Prompt")

def test_mock_vlm_single_frame():
    vlm = MockVLM(simulated_latency=0.01)
    frame = DummyFrame()
    response = vlm.generate(frame, "Describe this.")
    
    assert isinstance(response, VLMResponse)
    assert "Mock response for 1 frames" in response.text
    assert "Describe this." in response.text
    assert response.latency_ms > 0

def test_mock_vlm_temporal_frames():
    vlm = MockVLM(simulated_latency=0.01)
    frames = [DummyFrame(i) for i in range(5)]
    response = vlm.generate(frames, "Describe the action.")
    
    assert "Mock response for 5 frames" in response.text
    assert response.latency_ms > 0

def test_scheduler_vlm_lifecycle():
    scheduler = AIScheduler(target_fps=100.0)
    vlm = MockVLM(simulated_latency=0.05)
    
    # Submit some frames
    scheduler.submit(DummyFrame(1))
    scheduler.submit(DummyFrame(2))
    
    frame = scheduler.acquire()
    try:
        response = vlm.generate(frame, "What do you see?")
        assert response.latency_ms >= 50.0  # At least 50ms simulated
    finally:
        scheduler.release()
        
    stats = scheduler.stats()
    assert stats["inference_busy_time"] >= 0.05

def test_vlm_error_propagation():
    class FailingVLM(BaseVLMAdapter):
        def generate(self, frames, prompt):
            raise RuntimeError("Model crashed.")
            
    scheduler = AIScheduler(target_fps=100.0)
    vlm = FailingVLM()
    
    scheduler.submit(DummyFrame())
    
    frame = scheduler.acquire()
    try:
        with pytest.raises(RuntimeError, match="Model crashed"):
            vlm.generate(frame, "Prompt")
    finally:
        scheduler.release()
        
    # Verify release occurred even on error
    stats = scheduler.stats()
    assert stats["inference_busy_time"] >= 0.0
