import time
import torch
from velo.candidate import CandidateSelector
from velo.core import Frame

class MockFrame:
    def __init__(self, tensor):
        self._tensor = tensor
        
    def to_torch(self):
        return self._tensor

def run_candidate_benchmark():
    print(f"\n{'='*50}")
    print("BENCHMARK: CandidateSelector Performance")
    print(f"{'='*50}")
    
    selector = CandidateSelector(threshold=0.1, cooldown_ms=0.0)
    
    # Pre-allocate tensors (simulating NVDEC buffers)
    tensors = [
        torch.ones((480, 640, 4), dtype=torch.uint8) * 128,          # static
        torch.ones((480, 640, 4), dtype=torch.uint8) * 128,          # static (duplicate)
        torch.zeros((480, 640, 4), dtype=torch.uint8),               # massive change
        torch.ones((480, 640, 4), dtype=torch.uint8) * 200,          # medium change
    ]
    
    frames = [MockFrame(t) for t in tensors]
    
    # Warmup
    for f in frames:
        selector.analyze(f)
        
    iterations = 1000
    latencies = []
    
    evaluations = 0
    accepted = 0
    
    start_time = time.time()
    for i in range(iterations):
        # alternate between frames to simulate activity
        frame = frames[i % len(frames)]
        
        t0 = time.time()
        score = selector.analyze(frame)
        t1 = time.time()
        
        latencies.append((t1 - t0) * 1000)
        evaluations += 1
        if score.accepted:
            accepted += 1
            
    total_time = time.time() - start_time
    
    latencies.sort()
    avg_lat = sum(latencies) / len(latencies)
    p95_lat = latencies[int(len(latencies) * 0.95)]
    
    print(f"Total Evaluations: {evaluations}")
    print(f"Candidates Accepted: {accepted}")
    print(f"Candidates Rejected: {evaluations - accepted}")
    print(f"Acceptance Rate: {accepted / evaluations * 100:.1f}%")
    print(f"Average Latency: {avg_lat:.2f} ms")
    print(f"p95 Latency: {p95_lat:.2f} ms")
    print(f"Total Time: {total_time:.3f} s")
    print(f"{'='*50}")

if __name__ == "__main__":
    run_candidate_benchmark()
