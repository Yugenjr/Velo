"""
Velo Scheduler & Candidate Selection Benchmark

Benchmarks AIScheduler ring-buffer contention, candidate selection filtering,
and EWMA adaptive pacing latency overhead.
"""
import time
import sys
import os
import torch
import velo
from velo import AIScheduler, Frame
from benchmark_utils import save_benchmark_results


def create_mock_frame(ts, val=0):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    t = torch.full((480, 640, 3), val, dtype=torch.uint8, device=device)
    class MockCap:
        def __init__(self, t, timestamp):
            self.timestamp = timestamp
            self.shape = (480, 640, 3)
            self.__dlpack__ = t.__dlpack__
            self.__dlpack_device__ = t.__dlpack_device__
    return Frame(MockCap(t, ts), None)


def run_scheduler_benchmark(num_frames=1000):
    print(f"\n--- Running Scheduler Benchmark ({num_frames} frames) ---")

    # 1. Fixed Mode Benchmark
    sched_fixed = AIScheduler(target_fps=30, scene_aware=False, max_temporal_frames=60)
    t0 = time.time()
    for i in range(num_frames):
        f = create_mock_frame(ts=i/30.0, val=i % 255)
        sched_fixed.submit(f)
    t_fixed_submit = time.time() - t0

    # 2. Candidate-Aware Mode Benchmark
    sched_candidate = AIScheduler(target_fps=30, candidate_aware=True, max_temporal_frames=60)
    t0 = time.time()
    for i in range(num_frames):
        # Alternate identical frames with changed frames to exercise candidate rejection
        val = 100 if (i % 3 == 0) else 0
        f = create_mock_frame(ts=i/30.0, val=val)
        sched_candidate.submit(f)
    t_candidate_submit = time.time() - t0

    # 3. Adaptive Latency Feedback Benchmark
    sched_adaptive = AIScheduler(target_fps=30, adaptive=True, max_temporal_frames=60)
    t0 = time.time()
    for i in range(num_frames):
        sched_adaptive.record_inference(latency_ms=25.0 + (i % 10))
    t_adaptive_record = time.time() - t0

    stats_fixed = sched_fixed.stats()
    stats_cand = sched_candidate.stats()
    stats_adapt = sched_adaptive.stats()

    metrics = {
        "num_frames": num_frames,
        "fixed_submit_time_s": round(t_fixed_submit, 4),
        "fixed_submit_rate_fps": round(num_frames / max(0.0001, t_fixed_submit), 2),
        "candidate_submit_time_s": round(t_candidate_submit, 4),
        "candidate_submit_rate_fps": round(num_frames / max(0.0001, t_candidate_submit), 2),
        "candidates_evaluated": stats_cand.get("candidates_evaluated", 0),
        "candidates_accepted": stats_cand.get("candidates_accepted", 0),
        "candidates_rejected": stats_cand.get("candidates_rejected", 0),
        "candidate_acceptance_rate": round(stats_cand.get("candidate_acceptance_rate", 0.0), 3),
        "adaptive_record_time_s": round(t_adaptive_record, 4),
        "adaptive_record_rate_ops": round(num_frames / max(0.0001, t_adaptive_record), 2),
        "adaptive_target_fps": stats_adapt.get("target_fps", 0.0),
    }

    print(f"Fixed Submit Throughput: {metrics['fixed_submit_rate_fps']} ops/sec")
    print(f"Candidate Submit Throughput: {metrics['candidate_submit_rate_fps']} ops/sec (Rejection rate: {1.0 - metrics['candidate_acceptance_rate']:.2%})")
    print(f"Adaptive Feedback Rate: {metrics['adaptive_record_rate_ops']} updates/sec")

    config = {
        "num_frames": num_frames,
        "max_buffered_frames": 60,
    }
    save_benchmark_results("scheduler", config, metrics)
    return metrics


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    run_scheduler_benchmark(num_frames=n)
