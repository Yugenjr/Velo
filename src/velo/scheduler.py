import time
import threading
import collections


class SchedulerClosedError(Exception):
    """Raised when attempting to get a frame from a closed scheduler."""
    pass


class AIScheduler:
    """
    An AI-aware frame scheduling layer.
    
    Bridges high-framerate media pipelines (e.g., 30 FPS video) with slower
    AI inference workloads (e.g., 5 FPS Vision-Language Models).

    Enforces a `target_fps` by strictly pacing consumer reads and limits
    memory buffering by dropping stale frames via a bounded queue.
    """

    def __init__(self, target_fps: float = 5.0, max_queue: int = 2, strategy: str = "latest"):
        """
        Args:
            target_fps: The maximum rate (frames per second) at which the scheduler will yield frames.
            max_queue: Maximum number of frames to buffer before applying the drop strategy.
            strategy: The dropping strategy to apply when the queue is full. Currently supports "latest".
        """
        self.target_fps = target_fps
        self.max_queue = max_queue
        if strategy != "latest":
            raise ValueError(f"Unsupported strategy '{strategy}'. Only 'latest' is currently supported.")
        self.strategy = strategy
        
        self._queue = collections.deque(maxlen=max_queue)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._closed = False
        
        # Statistics
        self._frames_received = 0
        self._frames_processed = 0
        self._frames_dropped = 0
        self._total_queue_depth = 0
        self._total_frame_age = 0.0
        self._start_time = None
        self._last_yield_time = 0.0

    def submit(self, frame):
        """
        Submit a new frame to the scheduler.
        
        If the queue is at `max_queue`, the oldest unconsumed frame is dropped.
        """
        with self._lock:
            if self._closed:
                return
            
            if self._start_time is None:
                self._start_time = time.time()
                
            self._frames_received += 1
            
            if len(self._queue) == self.max_queue:
                self._queue.popleft()
                self._frames_dropped += 1
                
            self._queue.append((frame, time.time()))
            self._total_queue_depth += len(self._queue)
            self._cond.notify_all()

    def next(self):
        """
        Block until a frame is available and `1/target_fps` seconds have elapsed
        since the last yielded frame.
        
        Returns:
            The scheduled Frame.
            
        Raises:
            SchedulerClosedError: if the scheduler is closed and the queue is empty.
        """
        with self._cond:
            while len(self._queue) == 0 and not self._closed:
                self._cond.wait()
                
            if self._closed and len(self._queue) == 0:
                raise SchedulerClosedError("Scheduler is closed.")
                
            frame, enqueue_time = self._queue.popleft()
            
        # Outside the lock, enforce pacing
        now = time.time()
        elapsed_since_last_yield = now - self._last_yield_time
        sleep_time = (1.0 / self.target_fps) - elapsed_since_last_yield
        
        if sleep_time > 0:
            time.sleep(sleep_time)
            now = time.time()
            
        with self._lock:
            self._last_yield_time = now
            self._frames_processed += 1
            self._total_frame_age += (now - enqueue_time)
            
        return frame
        
    def close(self):
        """
        Signal the scheduler to shut down. Remaining buffered frames can still be consumed.
        """
        with self._cond:
            self._closed = True
            self._cond.notify_all()
            
    def stats(self) -> dict:
        """
        Return performance metrics and scheduling statistics.
        """
        with self._lock:
            avg_queue = self._total_queue_depth / max(1, self._frames_received)
            avg_age = self._total_frame_age / max(1, self._frames_processed)
            runtime = time.time() - self._start_time if self._start_time else 0
            effective_fps = self._frames_processed / runtime if runtime > 0 else 0.0
            
            return {
                "frames_received": self._frames_received,
                "frames_processed": self._frames_processed,
                "frames_dropped": self._frames_dropped,
                "average_queue_depth": avg_queue,
                "average_frame_age": avg_age,
                "effective_fps": effective_fps,
            }
