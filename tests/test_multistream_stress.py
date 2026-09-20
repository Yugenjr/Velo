import pytest
import asyncio
import time
import threading
import gc
import psutil
import torch
from typing import List

import velo
from velo import (
    MultiStreamPipeline,
    MockVLM,
    AIScheduler,
    AudioScheduler,
    TemporalFusion,
    Frame,
    AudioChunk,
    RuntimeMetrics,
)


class DeterministicStressSource:
    """Bounded, high-rate synthetic stream source for stress validation."""
    def __init__(self, stream_id: str, frame_count: int = 100, shape=(720, 1280, 3)):
        self.stream_id = stream_id
        self.frame_count = frame_count
        self.current_idx = 0
        self.shape = shape
        self._closed = False
        self._lock = threading.Lock()
        
        # Pre-allocate one resident tensor on CUDA to simulate zero-copy NVDEC output
        if torch.cuda.is_available():
            self._device_tensor = torch.zeros(shape, device="cuda", dtype=torch.uint8)
        else:
            self._device_tensor = torch.zeros(shape, dtype=torch.uint8)

    def next(self) -> Frame:
        with self._lock:
            if self._closed or self.current_idx >= self.frame_count:
                raise velo.StreamClosedError("Stream completed")
            time.sleep(0.002)
            self.current_idx += 1
            ts = self.current_idx * 0.033

            tensor_ref = self._device_tensor
            class Capsule:
                shape = (720, 1280, 3)
                timestamp = ts
                def __dlpack__(self, stream=None): return tensor_ref.__dlpack__()
                def __dlpack_device__(self): return tensor_ref.__dlpack_device__()

            frame = Frame(Capsule(), None)
            frame.arrival_time = time.time()
            return frame

    def close(self):
        with self._lock:
            self._closed = True


def get_process_memory_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def get_cuda_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


@pytest.mark.asyncio
async def test_stress_2_streams():
    """Stress test 2 streams under rapid ingestion and verify bounded memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cpu_mem_before = get_process_memory_mb()
    gpu_mem_before = get_cuda_memory_mb()

    vlm = MockVLM(simulated_latency=0.001)
    pipeline = MultiStreamPipeline(vlm=vlm)

    for i in range(2):
        sid = f"stress-2-{i}"
        src = DeterministicStressSource(sid, frame_count=60)
        pipeline.add_stream(sid, src, target_fps=30.0)

    await pipeline.start()

    inferences = 0
    async for _ in pipeline.run_inference("stress 2"):
        inferences += 1

    await pipeline.stop()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cpu_mem_after = get_process_memory_mb()
    gpu_mem_after = get_cuda_memory_mb()

    assert inferences > 0
    # Strict bounds: Memory growth must be bounded (<150MB CPU, <100MB GPU)
    assert (cpu_mem_after - cpu_mem_before) < 150.0
    assert (gpu_mem_after - gpu_mem_before) < 100.0


@pytest.mark.asyncio
async def test_stress_4_streams():
    """Stress test 4 streams under rapid ingestion and verify bounded memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cpu_mem_before = get_process_memory_mb()
    gpu_mem_before = get_cuda_memory_mb()

    vlm = MockVLM(simulated_latency=0.0005)
    pipeline = MultiStreamPipeline(vlm=vlm)

    for i in range(4):
        sid = f"stress-4-{i}"
        src = DeterministicStressSource(sid, frame_count=60)
        pipeline.add_stream(sid, src, target_fps=30.0)

    await pipeline.start()

    inferences = 0
    async for _ in pipeline.run_inference("stress 4"):
        inferences += 1

    await pipeline.stop()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cpu_mem_after = get_process_memory_mb()
    gpu_mem_after = get_cuda_memory_mb()

    assert inferences > 0
    assert (cpu_mem_after - cpu_mem_before) < 200.0
    assert (gpu_mem_after - gpu_mem_before) < 150.0


@pytest.mark.asyncio
async def test_stress_8_streams():
    """Stress test 8 streams (if hardware headroom permits)."""
    gpu_info = RuntimeMetrics.get_gpu_metrics()
    if gpu_info.get("available") and gpu_info.get("free_memory_mb", 0) < 150:
        pytest.skip("Insufficient VRAM for 8-stream stress benchmark")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cpu_mem_before = get_process_memory_mb()
    gpu_mem_before = get_cuda_memory_mb()

    vlm = MockVLM(simulated_latency=0.0002)
    pipeline = MultiStreamPipeline(vlm=vlm)

    for i in range(8):
        sid = f"stress-8-{i}"
        src = DeterministicStressSource(sid, frame_count=40)
        pipeline.add_stream(sid, src, target_fps=20.0)

    await pipeline.start()

    inferences = 0
    async for _ in pipeline.run_inference("stress 8"):
        inferences += 1

    await pipeline.stop()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cpu_mem_after = get_process_memory_mb()
    gpu_mem_after = get_cuda_memory_mb()

    assert inferences > 0
    assert (cpu_mem_after - cpu_mem_before) < 300.0
    assert (gpu_mem_after - gpu_mem_before) < 200.0
