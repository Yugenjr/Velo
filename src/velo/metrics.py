import time
import threading
from collections import deque
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


class RollingStats:
    """
    A thread-safe, bounded rolling window statistics tracker.
    Computes mean, median, p95, p99, min, max, and latest value without unbounded memory growth.
    """
    def __init__(self, maxlen: int = 1000):
        self.maxlen = maxlen
        self._samples: deque = deque(maxlen=maxlen)
        self._total_count = 0
        self._sum = 0.0
        self._min = float('inf')
        self._max = float('-inf')
        self._lock = threading.Lock()

    def update(self, value: float):
        """Record a new observation."""
        with self._lock:
            self._samples.append(value)
            self._total_count += 1
            self._sum += value
            if value < self._min:
                self._min = value
            if value > self._max:
                self._max = value

    def snapshot(self) -> Dict[str, float]:
        """Compute summary statistics for the rolling window."""
        with self._lock:
            count = len(self._samples)
            if count == 0:
                return {
                    "count": 0,
                    "mean": 0.0,
                    "median": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "latest": 0.0,
                }

            sorted_samples = sorted(self._samples)
            mean_val = sum(sorted_samples) / count
            median_val = sorted_samples[count // 2]
            p95_idx = min(count - 1, int(count * 0.95))
            p99_idx = min(count - 1, int(count * 0.99))
            p95_val = sorted_samples[p95_idx]
            p99_val = sorted_samples[p99_idx]
            min_val = sorted_samples[0]
            max_val = sorted_samples[-1]
            latest_val = self._samples[-1]

            return {
                "count": self._total_count,
                "window_size": count,
                "mean": round(mean_val, 3),
                "median": round(median_val, 3),
                "p95": round(p95_val, 3),
                "p99": round(p99_val, 3),
                "min": round(min_val, 3),
                "max": round(max_val, 3),
                "latest": round(latest_val, 3),
            }


class RuntimeMetrics:
    """
    Lightweight, low-overhead metrics collector for Velo.
    """
    def __init__(self):
        self.start_time = time.time()
        self._lock = threading.Lock()

        # Latency statistics
        self.inference_latency = RollingStats(maxlen=1000)
        self.preprocessing_latency = RollingStats(maxlen=1000)
        self.end_to_end_latency = RollingStats(maxlen=1000)
        self.fusion_lookup_latency = RollingStats(maxlen=1000)
        self.timestamp_skew = RollingStats(maxlen=1000)

        # Counters & Gauges
        self.inference_count = 0
        self.inference_errors = 0
        self.worker_errors = 0
        self.fusion_queries = 0

        # Rate calculations
        self._last_fps_calc_time = time.time()
        self._last_frames_processed = 0
        self._current_fps = 0.0

    def record_inference(
        self,
        inference_latency_ms: float,
        preprocessing_latency_ms: float = 0.0,
        end_to_end_latency_ms: Optional[float] = None,
    ):
        """Record inference timing metrics."""
        self.inference_latency.update(inference_latency_ms)
        if preprocessing_latency_ms > 0:
            self.preprocessing_latency.update(preprocessing_latency_ms)
        if end_to_end_latency_ms is not None:
            self.end_to_end_latency.update(end_to_end_latency_ms)
        with self._lock:
            self.inference_count += 1

    def record_inference_error(self):
        with self._lock:
            self.inference_errors += 1

    def record_worker_error(self):
        with self._lock:
            self.worker_errors += 1

    def record_fusion(self, lookup_latency_ms: float, skew_ms: Optional[float] = None):
        """Record fusion context lookup performance."""
        self.fusion_lookup_latency.update(lookup_latency_ms)
        if skew_ms is not None:
            self.timestamp_skew.update(skew_ms)
        with self._lock:
            self.fusion_queries += 1

    @staticmethod
    def get_gpu_metrics() -> Dict[str, Any]:
        """Safely retrieve GPU memory usage without hard dependencies."""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
                max_allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                device_name = torch.cuda.get_device_name(device)
                return {
                    "available": True,
                    "device": device_name,
                    "memory_allocated_mb": round(allocated, 2),
                    "memory_reserved_mb": round(reserved, 2),
                    "max_memory_allocated_mb": round(max_allocated, 2),
                }
        except Exception:
            pass

        return {
            "available": False,
            "device": None,
            "memory_allocated_mb": 0.0,
            "memory_reserved_mb": 0.0,
            "max_memory_allocated_mb": 0.0,
        }

    def snapshot(
        self,
        pipeline_state: str = "UNKNOWN",
        video_scheduler_stats: Optional[Dict[str, Any]] = None,
        audio_scheduler_stats: Optional[Dict[str, Any]] = None,
        fusion_stats: Optional[Dict[str, Any]] = None,
        inference_queue_depth: int = 0,
    ) -> Dict[str, Any]:
        """Return a complete, immutable snapshot of current runtime metrics."""
        now = time.time()
        uptime_s = round(now - self.start_time, 2)

        v_stats = video_scheduler_stats or {}
        a_stats = audio_scheduler_stats or {}
        f_stats = fusion_stats or {}

        return {
            "system": {
                "pipeline_state": pipeline_state,
                "uptime_s": uptime_s,
                "worker_errors": self.worker_errors,
            },
            "video": {
                "frames_received": v_stats.get("frames_received", 0),
                "frames_processed": v_stats.get("frames_processed", 0),
                "frames_dropped": v_stats.get("frames_dropped", 0),
                "frames_dropped_backpressure": v_stats.get("frames_dropped_backpressure", 0),
                "frames_dropped_stale": v_stats.get("frames_dropped_stale", 0),
                "frames_skipped_scene": v_stats.get("frames_skipped_scene", 0),
                "candidates_evaluated": v_stats.get("candidates_evaluated", 0),
                "candidates_accepted": v_stats.get("candidates_accepted", 0),
                "candidates_rejected": v_stats.get("candidates_rejected", 0),
                "candidate_acceptance_rate": v_stats.get("candidate_acceptance_rate", 0.0),
                "current_queue_depth": v_stats.get("current_queue_depth", 0),
                "max_queue_depth": v_stats.get("max_queue_depth", 0),
                "effective_fps": v_stats.get("effective_inference_fps", 0.0),
                "target_fps": v_stats.get("target_fps", 0.0),
                "adaptive_enabled": v_stats.get("adaptive_enabled", False),
            },
            "audio": {
                "chunks_received": a_stats.get("chunks_received", 0),
                "chunks_processed": a_stats.get("chunks_processed", 0),
                "chunks_dropped": a_stats.get("chunks_dropped", 0),
                "buffered_duration_s": a_stats.get("buffered_duration", 0.0),
                "average_chunk_latency_s": a_stats.get("average_chunk_latency", 0.0),
            },
            "inference": {
                "inference_count": self.inference_count,
                "inference_errors": self.inference_errors,
                "inference_queue_depth": inference_queue_depth,
                "inference_latency_ms": self.inference_latency.snapshot(),
                "preprocessing_latency_ms": self.preprocessing_latency.snapshot(),
                "end_to_end_latency_ms": self.end_to_end_latency.snapshot(),
            },
            "fusion": {
                "fusion_queries": self.fusion_queries,
                "lookup_latency_ms": self.fusion_lookup_latency.snapshot(),
                "timestamp_skew_ms": self.timestamp_skew.snapshot(),
                "video_buffer_size": f_stats.get("video_buffer_size", 0),
                "audio_buffer_size": f_stats.get("audio_buffer_size", 0),
            },
            "gpu": self.get_gpu_metrics(),
        }
