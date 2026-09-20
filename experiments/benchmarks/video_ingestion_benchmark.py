"""
Velo Video Ingestion & Scheduling Benchmark

Benchmarks video frame ingestion rate, NVDEC/synthetic GPU transfer,
and temporal buffer throughput over a target duration.
"""
import time
import sys
import os
import torch
import velo
from velo import AIScheduler, StreamClosedError
from benchmark_utils import save_benchmark_results, get_system_metadata


class SyntheticVideoSource:
    def __init__(self, target_fps=30, total_frames=900):
        self.target_fps = target_fps
        self.total_frames = total_frames
        self.frame_idx = 0
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._tensor = torch.zeros((1080, 1920, 3), dtype=torch.uint8, device=self.device)

    def next_frame(self):
        if self.frame_idx >= self.total_frames:
            raise StreamClosedError("End of benchmark video stream")
        ts = self.frame_idx / self.target_fps
        self.frame_idx += 1

        t = self._tensor
        class MockCapsule:
            def __init__(self, t, timestamp):
                self.timestamp = timestamp
                self.shape = (1080, 1920, 3)
                self.__dlpack__ = t.__dlpack__
                self.__dlpack_device__ = t.__dlpack_device__

        return MockCapsule(t, ts), None

    def close(self):
        pass


def run_video_benchmark(target_fps=30, duration_s=10.0):
    total_frames = int(target_fps * duration_s)
    source = SyntheticVideoSource(target_fps=target_fps, total_frames=total_frames)
    stream = velo.Stream(source)
    scheduler = AIScheduler(target_fps=target_fps, max_temporal_frames=60)

    print(f"\n--- Running Video Ingestion Benchmark ({duration_s}s @ {target_fps} FPS, {total_frames} frames) ---")
    t0 = time.time()
    frames_submitted = 0
    frames_acquired = 0

    # Ingestion loop
    while True:
        try:
            frame = stream.next()
            scheduler.submit(frame)
            frames_submitted += 1
        except StreamClosedError:
            break

    t_ingest_done = time.time()
    ingest_elapsed = t_ingest_done - t0

    # Close scheduler and drain available frames
    scheduler.close()
    while True:
        try:
            frame = scheduler.acquire()
            frames_acquired += 1
            scheduler.release()
        except Exception:
            break

    t_total_done = time.time()
    total_elapsed = t_total_done - t0

    stats = scheduler.stats()
    metrics = {
        "duration_s": round(duration_s, 2),
        "total_elapsed_s": round(total_elapsed, 4),
        "frames_submitted": frames_submitted,
        "frames_acquired": frames_acquired,
        "ingestion_throughput_fps": round(frames_submitted / max(0.001, ingest_elapsed), 2),
        "effective_fps": round(frames_acquired / max(0.001, total_elapsed), 2),
        "frames_dropped": stats.get("frames_dropped", 0),
        "frames_dropped_backpressure": stats.get("frames_dropped_backpressure", 0),
        "frames_dropped_stale": stats.get("frames_dropped_stale", 0),
        "average_frame_age_s": round(stats.get("average_frame_age", 0.0), 4),
    }

    print(f"Submitted: {frames_submitted} frames in {ingest_elapsed:.3f}s ({metrics['ingestion_throughput_fps']} FPS)")
    print(f"Acquired: {frames_acquired} frames | Dropped: {metrics['frames_dropped']}")

    config = {
        "target_fps": target_fps,
        "duration_s": duration_s,
        "resolution": "1920x1080",
    }
    save_benchmark_results("video_ingestion", config, metrics)
    return metrics


if __name__ == "__main__":
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    run_video_benchmark(target_fps=30, duration_s=dur)
