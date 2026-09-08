import pytest
import torch
import time
from velo.candidate import CandidateSelector, CandidateScore
from velo.core import Frame

class MockFrame:
    def __init__(self, tensor):
        self._tensor = tensor
    def to_torch(self):
        return self._tensor

def test_candidate_selector_initialization():
    selector = CandidateSelector(threshold=0.5, cooldown_ms=500.0, enabled=True)
    assert selector.threshold == 0.5
    assert selector.cooldown_ms == 500.0
    assert selector.enabled is True

def test_candidate_selector_disabled():
    selector = CandidateSelector(enabled=False)
    frame = MockFrame(None)
    score = selector.analyze(frame)
    assert score.accepted is True
    assert score.reason == "accepted"

def test_first_frame_novelty():
    selector = CandidateSelector()
    
    tensor = (torch.rand((480, 640, 4), dtype=torch.float32) * 255).to(torch.uint8)
    frame = MockFrame(tensor)
    
    score = selector.analyze(frame)
    assert score.accepted is True
    assert score.reason == "novelty"
    assert selector.reference_tensor is not None

def test_identical_frame_rejection():
    selector = CandidateSelector(threshold=0.1, cooldown_ms=0.0) # No cooldown
    
    tensor = torch.ones((480, 640, 4), dtype=torch.uint8) * 128
    frame = MockFrame(tensor)
    
    # First is novelty
    score1 = selector.analyze(frame)
    assert score1.accepted is True
    
    # Second is identical
    score2 = selector.analyze(frame)
    assert score2.accepted is False
    assert score2.reason == "low_information"
    assert score2.score == 0.0

def test_cooldown_lockout():
    selector = CandidateSelector(threshold=0.0, cooldown_ms=10000.0) # Huge cooldown
    
    tensor1 = (torch.rand((480, 640, 4), dtype=torch.float32) * 255).to(torch.uint8)
    frame1 = MockFrame(tensor1)
    
    tensor2 = (torch.rand((480, 640, 4), dtype=torch.float32) * 255).to(torch.uint8)
    frame2 = MockFrame(tensor2)
    
    score1 = selector.analyze(frame1)
    assert score1.accepted is True
    
    score2 = selector.analyze(frame2)
    assert score2.accepted is False
    assert score2.reason == "cooldown"

def test_significant_change():
    selector = CandidateSelector(threshold=0.1, cooldown_ms=0.0)
    
    # Dark frame
    tensor1 = torch.zeros((480, 640, 4), dtype=torch.uint8)
    frame1 = MockFrame(tensor1)
    
    # Bright noise frame (high spatial variance and change)
    tensor2 = (torch.rand((480, 640, 4), dtype=torch.float32) * 255).to(torch.uint8)
    frame2 = MockFrame(tensor2)
    
    score1 = selector.analyze(frame1)
    assert score1.accepted is True
    
    score2 = selector.analyze(frame2)
    assert score2.accepted is True
    assert score2.reason == "scene_change"
    assert score2.score > 0.1
