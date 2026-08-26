import pytest
import asyncio
import threading
from velo.pipeline import AIPipeline, PipelineState
from velo.scheduler import AIScheduler
from velo.vlm import MockVLM, BaseVLMAdapter, VLMResponse
from velo.exceptions import StreamClosedError
from velo.core import Frame

class MockStream:
    def __init__(self, num_frames=10, error_at_end=True, simulated_latency=0.01):
        self.num_frames = num_frames
        self.error_at_end = error_at_end
        self.simulated_latency = simulated_latency
        self.frames_yielded = 0
        self.is_closed = False

    def next(self):
        import time
        if self.is_closed:
            raise StreamClosedError("Stream is closed")
        if self.frames_yielded >= self.num_frames:
            if self.error_at_end:
                raise StreamClosedError("End of mock stream")
            else:
                time.sleep(1.0)
                raise StreamClosedError("Timeout")
        
        time.sleep(self.simulated_latency)
        self.frames_yielded += 1
        return "mock_frame"

    def close(self):
        self.is_closed = True

class CrashVLM(BaseVLMAdapter):
    def generate(self, frames, prompt):
        raise ValueError("VLM Crashed!")

@pytest.mark.asyncio
async def test_pipeline_lifecycle():
    stream = MockStream(num_frames=10)
    scheduler = AIScheduler(target_fps=30)
    vlm = MockVLM(simulated_latency=0.01)
    
    pipeline = AIPipeline(stream, scheduler, vlm)
    assert pipeline.state == PipelineState.CREATED
    
    await pipeline.start()
    assert pipeline.state == PipelineState.RUNNING
    
    await pipeline.stop()
    assert pipeline.state == PipelineState.STOPPED
    
    # Idempotent stop
    await pipeline.stop()
    assert pipeline.state == PipelineState.STOPPED

@pytest.mark.asyncio
async def test_invalid_transitions():
    stream = MockStream()
    pipeline = AIPipeline(stream, AIScheduler(target_fps=30), MockVLM())
    
    await pipeline.start()
    
    with pytest.raises(RuntimeError, match="Cannot start pipeline from state"):
        await pipeline.start()
        
    await pipeline.stop()

@pytest.mark.asyncio
async def test_pipeline_inference():
    stream = MockStream(num_frames=50) # Fast 100 FPS source
    scheduler = AIScheduler(target_fps=10) # Schedule 10 FPS
    vlm = MockVLM(simulated_latency=0.05) # 20 FPS VLM limit
    
    pipeline = AIPipeline(stream, scheduler, vlm)
    await pipeline.start()
    
    results = []
    
    # We will cancel this after we collect 3 results
    async def consumer():
        async for result in pipeline.run_inference("test"):
            results.append(result)
            if len(results) == 3:
                break
                
    await consumer()
    
    assert len(results) == 3
    assert "Mock response" in results[0].text
    
    await pipeline.stop()

@pytest.mark.asyncio
async def test_stream_closure_cascades():
    stream = MockStream(num_frames=5, error_at_end=True)
    scheduler = AIScheduler(target_fps=100)
    vlm = MockVLM()
    
    pipeline = AIPipeline(stream, scheduler, vlm)
    await pipeline.start()
    
    results = []
    async for result in pipeline.run_inference("test"):
        results.append(result)
        
    # The async generator should naturally exit when the stream ends and scheduler closes
    assert pipeline.state == PipelineState.STOPPED
    assert len(results) > 0

@pytest.mark.asyncio
async def test_vlm_exception_propagation():
    stream = MockStream(num_frames=50)
    scheduler = AIScheduler(target_fps=30)
    vlm = CrashVLM()
    
    pipeline = AIPipeline(stream, scheduler, vlm)
    await pipeline.start()
    
    with pytest.raises(ValueError, match="VLM Crashed!"):
        async for result in pipeline.run_inference("test"):
            pass
            
    assert pipeline.state == PipelineState.FAILED

@pytest.mark.asyncio
async def test_single_consumer_enforcement():
    stream = MockStream(num_frames=50)
    pipeline = AIPipeline(stream, AIScheduler(target_fps=30), MockVLM())
    await pipeline.start()
    
    async def first_consumer():
        async for r in pipeline.run_inference("first"):
            await asyncio.sleep(0.1)
            
    task = asyncio.create_task(first_consumer())
    await asyncio.sleep(0.01) # let it start
    
    with pytest.raises(RuntimeError, match="Only one active inference consumer"):
        async for r in pipeline.run_inference("second"):
            pass
            
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
        
    await pipeline.stop()
