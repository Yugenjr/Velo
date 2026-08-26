import time
import threading
from velo.scheduler import AIScheduler

class MockFrame:
    def __init__(self, index):
        self.index = index
        self.timestamp = time.time()

def run_benchmark(scenario_name: str, simulated_latency: float, duration: float = 5.0, adaptive: bool = True):
    print(f"\n{'='*50}")
    print(f"BENCHMARK: {scenario_name}")
    print(f"Adaptive: {adaptive}, Simulated Inference Latency: {simulated_latency*1000:.1f}ms")
    print(f"{'='*50}")
    
    # 30 FPS ingress, 10 FPS initial target
    scheduler = AIScheduler(
        target_fps=10.0,
        adaptive=adaptive,
        min_fps=1.0,
        max_fps=30.0,
        latency_budget_ms=200.0
    )
    
    ingress_active = True
    
    def ingress_worker():
        frame_idx = 0
        while ingress_active:
            scheduler.submit(MockFrame(frame_idx))
            frame_idx += 1
            time.sleep(1/30.0) # 30 FPS
            
    t_ingress = threading.Thread(target=ingress_worker, daemon=True)
    t_ingress.start()
    
    start_time = time.time()
    frames_inferred = 0
    latencies = []
    
    while time.time() - start_time < duration:
        try:
            frame = scheduler.acquire()
            # Simulate inference
            t0 = time.time()
            time.sleep(simulated_latency)
            latency_ms = (time.time() - t0) * 1000
            scheduler.record_inference(latency_ms)
            
            latencies.append(latency_ms)
            frames_inferred += 1
            scheduler.release()
        except Exception as e:
            print(f"Error: {e}")
            break
            
    ingress_active = False
    scheduler.close()
    t_ingress.join(timeout=1.0)
    
    stats = scheduler.stats()
    
    avg_lat = sum(latencies)/len(latencies) if latencies else 0
    latencies.sort()
    p95_lat = latencies[int(len(latencies)*0.95)] if latencies else 0
    
    print(f"Frames Ingressed: {stats['frames_received']}")
    print(f"Frames Inferred: {stats['frames_processed']}")
    print(f"Frames Dropped: {stats['frames_dropped']}")
    print(f"Average Inference Latency: {avg_lat:.1f} ms")
    print(f"p95 Inference Latency: {p95_lat:.1f} ms")
    print(f"Effective Inference FPS: {stats['effective_inference_fps']:.2f}")
    if adaptive:
        print(f"Final Target FPS: {stats['target_fps']:.2f}")
    print(f"Scheduler Utilization (Busy Time): {stats['inference_busy_time']:.2f} s")
    print(f"{'='*50}")

if __name__ == "__main__":
    print("Starting Adaptive Scheduler Benchmarks...")
    
    # Baseline: Fixed scheduler, slow model
    run_benchmark("Fixed Mode (Slow Model)", simulated_latency=0.3, adaptive=False)
    
    # Adaptive: Slow model (should scale FPS down)
    run_benchmark("Adaptive Mode (Slow Model)", simulated_latency=0.3, adaptive=True)
    
    # Adaptive: Fast model (should scale FPS up)
    run_benchmark("Adaptive Mode (Fast Model)", simulated_latency=0.05, adaptive=True)
