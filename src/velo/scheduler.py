import time
import threading
import collections


class SchedulerClosedError(Exception):
    """Raised when attempting to get a frame from a closed scheduler."""
    pass


class AIScheduler:
    """
    An AI-aware temporal scheduling layer.

    Bridges high-framerate media pipelines (e.g., 30 FPS video) with slower
    AI inference workloads (e.g., 5 FPS Vision-Language Models).

    Provides a configurable temporal buffer that retains recent visual context,
    enforces backpressure when the AI model is busy, and avoids unbounded queue
    growth. Preserves GPU frame identity.
    """

    def __init__(self,
                 target_fps: float = 5.0,
                 temporal_window: float = 5.0,
                 max_temporal_frames: int = 150,
                 scene_aware: bool = False,
                 scene_threshold: float = 0.05,
                 candidate_aware: bool = False,
                 candidate_threshold: float = 0.1,
                 candidate_cooldown_ms: float = 1000.0,
                 adaptive: bool = False,
                 min_fps: float = 1.0,
                 max_fps: float = 30.0,
                 latency_budget_ms: float = 250.0):
        """
        Args:
            target_fps: Maximum rate (frames per second) at which the scheduler yields frames.
            temporal_window: How many seconds of history to retain in the temporal buffer.
            max_temporal_frames: Maximum absolute number of frames to retain in the buffer.
            scene_aware: If True, uses GPU scene detection to skip visually static frames.
            scene_threshold: L1 difference threshold for scene detection.
            candidate_aware: If True, evaluates candidate quality before inference to skip low-info frames.
            candidate_threshold: Score threshold for the candidate selector.
            candidate_cooldown_ms: Minimum time in ms between accepted candidates.
            adaptive: If True, dynamically adjusts target_fps based on inference latency feedback.
            min_fps: Minimum FPS bound for adaptive mode.
            max_fps: Maximum FPS bound for adaptive mode.
            latency_budget_ms: Target inference latency budget.
        """
        self.target_fps = target_fps
        self.temporal_window = temporal_window
        self.max_temporal_frames = max_temporal_frames

        self.scene_aware = scene_aware
        if self.scene_aware:
            from .scene import SceneChangeDetector
            self.scene_detector = SceneChangeDetector(threshold=scene_threshold)

        self.candidate_aware = candidate_aware
        if self.candidate_aware:
            from .candidate import CandidateSelector
            self.candidate_selector = CandidateSelector(threshold=candidate_threshold, cooldown_ms=candidate_cooldown_ms)

        self.adaptive = adaptive
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.latency_budget_ms = latency_budget_ms
        self._ewma_latency = 0.0
        self._alpha = 0.2

        self._buffer = collections.deque()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._closed = False

        self._worker_busy = False
        self._worker_busy_start = None
        self._has_new_frame = False

        # Statistics
        self._frames_received = 0
        self._frames_processed = 0
        self._frames_dropped = 0
        self._frames_dropped_stale = 0
        self._frames_dropped_backpressure = 0
        self._frames_skipped_scene = 0

        self._candidates_evaluated = 0
        self._candidates_accepted = 0
        self._candidates_rejected = 0

        self._total_frame_age = 0.0
        self._inference_busy_time = 0.0
        self._start_time = None
        self._last_yield_time = 0.0
        self._max_queue_depth = 0

    def _evict_stale_frames(self, now: float):
        """Evict frames outside the temporal window or exceeding the max limit."""
        while self._buffer and (now - self._buffer[0][1]) > self.temporal_window:
            self._buffer.popleft()
            self._frames_dropped_stale += 1
            self._frames_dropped += 1

        while len(self._buffer) > self.max_temporal_frames:
            self._buffer.popleft()
            self._frames_dropped_stale += 1
            self._frames_dropped += 1

    def submit(self, frame):
        """
        Submit a new frame. Appends it to the temporal buffer.
        Records backpressure metrics if the inference worker hasn't consumed the previous latest frame.
        """
        now = time.time()
        with self._lock:
            if self._closed:
                return

            if self._start_time is None:
                self._start_time = now

            self._frames_received += 1

            if self._has_new_frame:
                self._frames_dropped_backpressure += 1
                self._frames_dropped += 1

            self._buffer.append((frame, now))
            self._max_queue_depth = max(self._max_queue_depth, len(self._buffer))

            self._evict_stale_frames(now)

            self._has_new_frame = True
            self._cond.notify_all()

    def acquire(self):
        """
        Block until a fresh frame is available, target FPS pacing is met,
        and (if scene_aware is True) a visual change is detected.
        Marks the inference worker as busy.

        Returns:
            The latest scheduled Frame.

        Raises:
            SchedulerClosedError: if the scheduler is closed.
        """
        while True:
            with self._cond:
                while not self._has_new_frame and not self._closed:
                    self._cond.wait()

                if self._closed and not self._has_new_frame:
                    raise SchedulerClosedError("Scheduler is closed.")

                frame, enqueue_time = self._buffer[-1]
                self._has_new_frame = False

            now = time.time()
            elapsed_since_last = now - self._last_yield_time
            sleep_time = (1.0 / self.target_fps) - elapsed_since_last

            if sleep_time > 0:
                time.sleep(sleep_time)
                now = time.time()

            # Evaluate scene change heuristic
            if self.scene_aware:
                result = self.scene_detector.update(frame)
                if not result.changed:
                    with self._lock:
                        self._frames_skipped_scene += 1
                    continue

            if self.candidate_aware:
                cand_score = self.candidate_selector.analyze(frame)
                with self._lock:
                    self._candidates_evaluated += 1
                    if cand_score.accepted:
                        self._candidates_accepted += 1
                    else:
                        self._candidates_rejected += 1

                if not cand_score.accepted:
                    continue

            # Frame is accepted for inference
            now = time.time()

            with self._lock:
                self._last_yield_time = now
                self._worker_busy = True
                self._worker_busy_start = now
                self._frames_processed += 1

                capture_ts = getattr(frame, "timestamp", 0.0)
                if capture_ts > 0:
                    age = now - capture_ts
                else:
                    age = now - enqueue_time

                self._total_frame_age += age

            return frame

    def release(self):
        """
        Mark the inference worker as idle.
        Must be called after processing the frame acquired from `acquire()`.
        """
        now = time.time()
        with self._lock:
            if self._worker_busy:
                self._inference_busy_time += (now - self._worker_busy_start)
                self._worker_busy = False
                self._worker_busy_start = None

    def record_inference(self, latency_ms: float):
        """
        Record the latency of a completed inference to drive adaptive scheduling.

        Args:
            latency_ms: The total inference latency in milliseconds.
        """
        with self._lock:
            if not self.adaptive:
                return

            if self._ewma_latency == 0.0:
                self._ewma_latency = latency_ms
            else:
                self._ewma_latency = (self._alpha * latency_ms) + ((1 - self._alpha) * self._ewma_latency)

            if self._ewma_latency > self.latency_budget_ms:
                self.target_fps = max(self.min_fps, self.target_fps * 0.9)
            elif self._ewma_latency < self.latency_budget_ms * 0.5:
                self.target_fps = min(self.max_fps, self.target_fps * 1.05)

    def next(self):
        """
        Legacy support for simple acquire-and-release.
        Useful when the caller does not block the thread during inference.
        """
        frame = self.acquire()
        self.release()
        return frame

    def snapshot(self, duration: float = 1.0, max_frames: int = 8) -> list:
        """
        Return a temporal sequence of recent frames from the buffer.

        Args:
            duration: Maximum age (in seconds) of frames to include.
            max_frames: Maximum number of frames to return.

        Returns:
            A list of Frame objects in chronological order (oldest first).
        """
        now = time.time()
        with self._lock:
            self._evict_stale_frames(now)

            result = []
            for frame, enqueue_time in reversed(self._buffer):
                if (now - enqueue_time) > duration:
                    break
                result.append(frame)
                if len(result) == max_frames:
                    break

            return list(reversed(result))

    def close(self):
        """Signal the scheduler to shut down."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def stats(self) -> dict:
        """Return temporal and backpressure statistics."""
        now = time.time()
        with self._lock:
            avg_age = self._total_frame_age / max(1, self._frames_processed)

            busy_time = self._inference_busy_time
            if self._worker_busy and self._worker_busy_start:
                busy_time += (now - self._worker_busy_start)

        # Effective fps based on time since first acquire
        if self._start_time and time.time() - self._start_time > 0:
            effective_fps = self._frames_processed / (time.time() - self._start_time)
        else:
            effective_fps = 0.0

        acc_rate = 0.0
        if self._candidates_evaluated > 0:
            acc_rate = self._candidates_accepted / self._candidates_evaluated

        with self._lock:
            return {
                "frames_received": self._frames_received,
                "frames_processed": self._frames_processed,
                "frames_dropped": self._frames_dropped,
                "frames_dropped_stale": self._frames_dropped_stale,
                "frames_dropped_backpressure": self._frames_dropped_backpressure,
                "frames_skipped_scene": self._frames_skipped_scene,
                "candidates_evaluated": self._candidates_evaluated,
                "candidates_accepted": self._candidates_accepted,
                "candidates_rejected": self._candidates_rejected,
                "candidate_acceptance_rate": acc_rate,
                "current_queue_depth": len(self._buffer),
                "max_queue_depth": self._max_queue_depth,
                "average_frame_age": avg_age,
                "inference_busy_time": busy_time,
                "effective_inference_fps": effective_fps,
                "adaptive_enabled": self.adaptive,
                "target_fps": self.target_fps,
                "ewma_latency_ms": self._ewma_latency,
            }
