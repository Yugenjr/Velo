"""
Velo Temporal Fusion & Correlation Benchmark

Benchmarks TemporalFusion ingestion rate, memory window eviction,
context lookup latency, and timestamp skew tracking.
"""
import time
import sys
import os
import torch
import velo
from velo import TemporalFusion, Frame, Transcript
from benchmark_utils import save_benchmark_results


def run_fusion_benchmark(num_observations=5000, num_queries=1000):
    print(f"\n--- Running Temporal Fusion Benchmark ({num_observations} observations, {num_queries} queries) ---")
    fusion = TemporalFusion(max_history_s=30.0)

    # Ingest video observations
    t0 = time.time()
    for i in range(num_observations):
        ts = i * 0.033 # 30 FPS
        class MockCap:
            def __init__(self, timestamp):
                self.timestamp = timestamp
                self.shape = (10, 10, 3)
        frame = Frame(MockCap(ts), None)
        fusion.add_video(frame, timestamp=ts)
    t_video_ingest = time.time() - t0

    # Ingest audio/transcript observations
    t0 = time.time()
    for i in range(num_observations // 10):
        start_ts = i * 0.33
        end_ts = start_ts + 0.30
        t = Transcript(text=f"transcript segment {i}", start_timestamp=start_ts, end_timestamp=end_ts, confidence=0.99)
        fusion.add_audio(t, timestamp=start_ts, duration=0.30)
    t_audio_ingest = time.time() - t0

    # Query context repeatedly across random window timestamps
    t0 = time.time()
    matched_video_total = 0
    matched_audio_total = 0
    for q in range(num_queries):
        query_ts = (q % 100) * 1.5
        ctx = fusion.context(timestamp=query_ts, window=1.5)
        matched_video_total += len(ctx.video)
        matched_audio_total += len(ctx.audio)
    t_queries = time.time() - t0

    stats = fusion.stats()
    metrics = {
        "num_video_observations": num_observations,
        "num_audio_observations": num_observations // 10,
        "num_queries": num_queries,
        "video_ingest_rate_ops": round(num_observations / max(0.0001, t_video_ingest), 2),
        "audio_ingest_rate_ops": round((num_observations // 10) / max(0.0001, t_audio_ingest), 2),
        "query_throughput_qps": round(num_queries / max(0.0001, t_queries), 2),
        "avg_lookup_latency_ms": stats.get("avg_lookup_latency_ms", 0.0),
        "avg_skew_ms": stats.get("avg_skew_ms", 0.0),
        "max_skew_ms": stats.get("max_skew_ms", 0.0),
        "avg_matched_video": round(matched_video_total / max(1, num_queries), 2),
        "avg_matched_audio": round(matched_audio_total / max(1, num_queries), 2),
    }

    print(f"Video Ingestion Rate: {metrics['video_ingest_rate_ops']} obs/sec")
    print(f"Audio Ingestion Rate: {metrics['audio_ingest_rate_ops']} obs/sec")
    print(f"Context Query Rate: {metrics['query_throughput_qps']} QPS (Avg lookup: {metrics['avg_lookup_latency_ms']} ms)")
    print(f"Average Timestamp Skew: {metrics['avg_skew_ms']} ms | Max Skew: {metrics['max_skew_ms']} ms")

    config = {
        "max_history_s": 30.0,
        "num_observations": num_observations,
        "num_queries": num_queries,
    }
    save_benchmark_results("fusion", config, metrics)
    return metrics


if __name__ == "__main__":
    n_obs = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    n_q = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    run_fusion_benchmark(num_observations=n_obs, num_queries=n_q)
