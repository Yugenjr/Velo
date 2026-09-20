"""
Example 04: Runtime Observability & Live Metrics Snapshotting

Demonstrates querying non-blocking, thread-safe runtime telemetry
from a live Velo pipeline instance.
"""
import velo
import json

def main():
    metrics = velo.RuntimeMetrics()
    
    # Record sample inference & fusion latencies
    metrics.record_inference(inference_latency_ms=12.4, preprocessing_latency_ms=1.1, end_to_end_latency_ms=25.0)
    metrics.record_fusion(lookup_latency_ms=0.035, skew_ms=45.0)
    
    # Capture immutable snapshot
    snapshot = metrics.snapshot(
        pipeline_state="RUNNING",
        video_scheduler_stats={"frames_received": 300, "frames_processed": 50, "frames_dropped": 250},
        audio_scheduler_stats={"chunks_received": 500, "chunks_processed": 50, "chunks_dropped": 0},
    )
    
    print("Velo Runtime Metrics Snapshot:")
    print(json.dumps(snapshot, indent=2))

if __name__ == "__main__":
    main()
