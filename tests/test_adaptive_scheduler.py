import pytest
import time
import threading
from velo.scheduler import AIScheduler

class MockFrame:
    def __init__(self, index, timestamp=None):
        self.index = index
        self.timestamp = timestamp or time.time()

def test_fixed_mode_unchanged():
    scheduler = AIScheduler(target_fps=10.0, adaptive=False)
    assert scheduler.target_fps == 10.0
    
    # Simulating long latency shouldn't change target_fps in fixed mode
    scheduler.record_inference(1000.0)
    assert scheduler.target_fps == 10.0

def test_adaptive_mode_construction():
    scheduler = AIScheduler(
        target_fps=5.0,
        adaptive=True,
        min_fps=2.0,
        max_fps=15.0,
        latency_budget_ms=200.0
    )
    assert scheduler.adaptive
    assert scheduler.min_fps == 2.0
    assert scheduler.max_fps == 15.0
    assert scheduler.latency_budget_ms == 200.0

def test_adaptive_latency_decrease_fps():
    scheduler = AIScheduler(
        target_fps=10.0,
        adaptive=True,
        latency_budget_ms=100.0,
        min_fps=1.0
    )
    
    # Simulate high latency (500ms > 100ms budget)
    scheduler.record_inference(500.0)
    
    # 500ms is recorded as first EWMA. It exceeds 100ms.
    # target_fps should drop by factor of 0.9.
    assert scheduler.target_fps == 9.0
    
    # Record again
    scheduler.record_inference(500.0)
    assert scheduler.target_fps == 8.1
    
def test_adaptive_latency_increase_fps():
    scheduler = AIScheduler(
        target_fps=5.0,
        adaptive=True,
        latency_budget_ms=200.0,
        max_fps=30.0
    )
    
    # Simulate very low latency (50ms < 100ms [budget/2])
    scheduler.record_inference(50.0)
    
    # Should increase by factor of 1.05
    assert scheduler.target_fps == 5.25

def test_adaptive_clamping():
    scheduler = AIScheduler(
        target_fps=10.0,
        adaptive=True,
        latency_budget_ms=100.0,
        max_fps=12.0,
        min_fps=8.0
    )
    
    # Push latency high to hit min_fps
    for _ in range(20):
        scheduler.record_inference(500.0)
        
    assert scheduler.target_fps == 8.0 # Clamped to min_fps
    
    # Push latency low to hit max_fps
    for _ in range(50):
        scheduler.record_inference(10.0)
        
    assert scheduler.target_fps == 12.0 # Clamped to max_fps

def test_adaptive_ewma_smoothing():
    scheduler = AIScheduler(
        target_fps=10.0,
        adaptive=True,
        latency_budget_ms=200.0
    )
    
    # Seed with good latency
    scheduler.record_inference(150.0)
    assert scheduler.target_fps == 10.0
    
    # One spike shouldn't immediately ruin the EWMA if alpha=0.2
    # New EWMA = (0.2 * 300) + (0.8 * 150) = 60 + 120 = 180 (still < 200)
    scheduler.record_inference(300.0)
    assert scheduler.target_fps == 10.0 # Remains unchanged because EWMA is 180

def test_adaptive_stats_reporting():
    scheduler = AIScheduler(target_fps=10.0, adaptive=True, latency_budget_ms=100.0)
    scheduler.record_inference(200.0)
    
    stats = scheduler.stats()
    assert stats["adaptive_enabled"] is True
    assert stats["target_fps"] == 9.0
    assert stats["ewma_latency_ms"] == 200.0
