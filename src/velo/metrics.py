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


class StreamMetrics:
    """
    Per-stream metrics collector tracking latency, throughput, queues, and drops.
    Supports both dictionary-like access and method access.
    """
    def __init__(self, stream_id: str = "default"):
        self.stream_id = stream_id
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

        # Stats state cache
        self.video_stats: Dict[str, Any] = {}
        self.audio_stats: Dict[str, Any] = {}
        self.fusion_stats: Dict[str, Any] = {}
        self.pipeline_state: str = "CREATED"
        self.inference_queue_depth: int = 0

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

    def update_state(
        self,
        pipeline_state: Optional[str] = None,
        video_scheduler_stats: Optional[Dict[str, Any]] = None,
        audio_scheduler_stats: Optional[Dict[str, Any]] = None,
        fusion_stats: Optional[Dict[str, Any]] = None,
        inference_queue_depth: Optional[int] = None,
    ):
        with self._lock:
            if pipeline_state is not None:
                self.pipeline_state = pipeline_state
            if video_scheduler_stats is not None:
                self.video_stats = video_scheduler_stats
            if audio_scheduler_stats is not None:
                self.audio_stats = audio_scheduler_stats
            if fusion_stats is not None:
                self.fusion_stats = fusion_stats
            if inference_queue_depth is not None:
                self.inference_queue_depth = inference_queue_depth

    def snapshot(
        self,
        pipeline_state: Optional[str] = None,
        video_scheduler_stats: Optional[Dict[str, Any]] = None,
        audio_scheduler_stats: Optional[Dict[str, Any]] = None,
        fusion_stats: Optional[Dict[str, Any]] = None,
        inference_queue_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return a complete, immutable snapshot of current stream metrics."""
        now = time.time()
        uptime_s = round(now - self.start_time, 2)

        with self._lock:
            p_state = pipeline_state if pipeline_state is not None else self.pipeline_state
            v_stats = video_scheduler_stats if video_scheduler_stats is not None else self.video_stats
            a_stats = audio_scheduler_stats if audio_scheduler_stats is not None else self.audio_stats
            f_stats = fusion_stats if fusion_stats is not None else self.fusion_stats
            q_depth = inference_queue_depth if inference_queue_depth is not None else self.inference_queue_depth

            return {
                "stream_id": self.stream_id,
                "system": {
                    "pipeline_state": p_state,
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
                    "inference_queue_depth": q_depth,
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
                "gpu": RuntimeMetrics.get_gpu_metrics(),
            }

    def __getitem__(self, item: str):
        return self.snapshot()[item]

    def get(self, key: str, default: Any = None) -> Any:
        return self.snapshot().get(key, default)


class RuntimeMetrics:
    """
    Lightweight, low-overhead metrics collector for single-stream and multi-stream Velo.
    """
    def __init__(self, default_stream_id: Optional[str] = "stream_0"):
        self.start_time = time.time()
        self._lock = threading.Lock()
        self._default_stream_id = default_stream_id
        if default_stream_id is not None:
            self._streams: Dict[str, StreamMetrics] = {
                default_stream_id: StreamMetrics(default_stream_id)
            }
            self._default_stream = self._streams[default_stream_id]
        else:
            self._streams: Dict[str, StreamMetrics] = {}
            self._default_stream = None

    def _get_default_stream(self) -> StreamMetrics:
        with self._lock:
            if self._default_stream is not None:
                return self._default_stream
            if self._streams:
                return next(iter(self._streams.values()))
            def_stream = StreamMetrics("default")
            self._streams["default"] = def_stream
            self._default_stream = def_stream
            return def_stream

    # Single-stream property delegates for 100% backwards compatibility
    @property
    def inference_latency(self) -> RollingStats:
        return self._get_default_stream().inference_latency

    @property
    def preprocessing_latency(self) -> RollingStats:
        return self._get_default_stream().preprocessing_latency

    @property
    def end_to_end_latency(self) -> RollingStats:
        return self._get_default_stream().end_to_end_latency

    @property
    def fusion_lookup_latency(self) -> RollingStats:
        return self._get_default_stream().fusion_lookup_latency

    @property
    def timestamp_skew(self) -> RollingStats:
        return self._get_default_stream().timestamp_skew

    @property
    def inference_count(self) -> int:
        return self._get_default_stream().inference_count

    @property
    def inference_errors(self) -> int:
        return self._get_default_stream().inference_errors

    @property
    def worker_errors(self) -> int:
        return self._get_default_stream().worker_errors

    @property
    def fusion_queries(self) -> int:
        return self._get_default_stream().fusion_queries

    def register_stream(self, stream_id: str) -> StreamMetrics:
        """Register a new stream metrics collector."""
        with self._lock:
            if stream_id not in self._streams:
                self._streams[stream_id] = StreamMetrics(stream_id)
            return self._streams[stream_id]

    def unregister_stream(self, stream_id: str):
        """Unregister a stream from active tracking."""
        with self._lock:
            if stream_id in self._streams and stream_id != self._default_stream_id:
                del self._streams[stream_id]

    def stream(self, stream_id: str) -> StreamMetrics:
        """Access metrics for a specific stream."""
        with self._lock:
            if stream_id not in self._streams:
                self._streams[stream_id] = StreamMetrics(stream_id)
            return self._streams[stream_id]

    def stream_metrics(self, stream_id: str) -> StreamMetrics:
        return self.stream(stream_id)

    def record_inference(
        self,
        inference_latency_ms: float,
        preprocessing_latency_ms: float = 0.0,
        end_to_end_latency_ms: Optional[float] = None,
        stream_id: Optional[str] = None,
    ):
        """Record inference timing metrics for default or target stream."""
        target = self.stream(stream_id) if stream_id else self._get_default_stream()
        target.record_inference(inference_latency_ms, preprocessing_latency_ms, end_to_end_latency_ms)

    def record_inference_error(self, stream_id: Optional[str] = None):
        target = self.stream(stream_id) if stream_id else self._get_default_stream()
        target.record_inference_error()

    def record_worker_error(self, stream_id: Optional[str] = None):
        target = self.stream(stream_id) if stream_id else self._get_default_stream()
        target.record_worker_error()

    def record_fusion(
        self,
        lookup_latency_ms: float,
        skew_ms: Optional[float] = None,
        stream_id: Optional[str] = None,
    ):
        """Record fusion context lookup performance."""
        target = self.stream(stream_id) if stream_id else self._get_default_stream()
        target.record_fusion(lookup_latency_ms, skew_ms)

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
                total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
                return {
                    "available": True,
                    "device": device_name,
                    "device_index": device,
                    "total_memory_mb": round(total_memory, 2),
                    "memory_allocated_mb": round(allocated, 2),
                    "memory_reserved_mb": round(reserved, 2),
                    "max_memory_allocated_mb": round(max_allocated, 2),
                    "free_memory_mb": round(total_memory - allocated, 2),
                }
        except Exception:
            pass

        return {
            "available": False,
            "device": None,
            "device_index": 0,
            "total_memory_mb": 0.0,
            "memory_allocated_mb": 0.0,
            "memory_reserved_mb": 0.0,
            "max_memory_allocated_mb": 0.0,
            "free_memory_mb": 0.0,
        }

    def global_snapshot(self) -> Dict[str, Any]:
        """Aggregate process-wide metrics across all active streams."""
        now = time.time()
        uptime_s = round(now - self.start_time, 2)

        with self._lock:
            stream_snapshots = {sid: s.snapshot() for sid, s in self._streams.items()}

        total_video_fps = sum(s["video"]["effective_fps"] for s in stream_snapshots.values())
        total_video_received = sum(s["video"]["frames_received"] for s in stream_snapshots.values())
        total_video_processed = sum(s["video"]["frames_processed"] for s in stream_snapshots.values())
        total_video_dropped = sum(s["video"]["frames_dropped"] for s in stream_snapshots.values())

        total_audio_received = sum(s["audio"]["chunks_received"] for s in stream_snapshots.values())
        total_audio_processed = sum(s["audio"]["chunks_processed"] for s in stream_snapshots.values())
        total_audio_dropped = sum(s["audio"]["chunks_dropped"] for s in stream_snapshots.values())

        total_inference_count = sum(s["inference"]["inference_count"] for s in stream_snapshots.values())
        total_inference_errors = sum(s["inference"]["inference_errors"] for s in stream_snapshots.values())
        total_worker_errors = sum(s["system"]["worker_errors"] for s in stream_snapshots.values())
        total_fusion_queries = sum(s["fusion"]["fusion_queries"] for s in stream_snapshots.values())

        return {
            "system": {
                "active_streams": len(stream_snapshots),
                "uptime_s": uptime_s,
                "total_worker_errors": total_worker_errors,
            },
            "streams": list(stream_snapshots.keys()),
            "video": {
                "total_fps": round(total_video_fps, 2),
                "total_frames_received": total_video_received,
                "total_frames_processed": total_video_processed,
                "total_frames_dropped": total_video_dropped,
            },
            "audio": {
                "total_chunks_received": total_audio_received,
                "total_chunks_processed": total_audio_processed,
                "total_chunks_dropped": total_audio_dropped,
            },
            "inference": {
                "total_inferences": total_inference_count,
                "total_errors": total_inference_errors,
            },
            "fusion": {
                "total_queries": total_fusion_queries,
            },
            "gpu": self.get_gpu_metrics(),
        }

    def snapshot(
        self,
        pipeline_state: str = "UNKNOWN",
        video_scheduler_stats: Optional[Dict[str, Any]] = None,
        audio_scheduler_stats: Optional[Dict[str, Any]] = None,
        fusion_stats: Optional[Dict[str, Any]] = None,
        inference_queue_depth: int = 0,
    ) -> Dict[str, Any]:
        """Return snapshot compatible with both single-stream and multi-stream clients."""
        return self._get_default_stream().snapshot(
            pipeline_state=pipeline_state,
            video_scheduler_stats=video_scheduler_stats,
            audio_scheduler_stats=audio_scheduler_stats,
            fusion_stats=fusion_stats,
            inference_queue_depth=inference_queue_depth,
        )


# Alias `metrics.global()` to `metrics.global_snapshot()` dynamically
setattr(RuntimeMetrics, "global", RuntimeMetrics.global_snapshot)
