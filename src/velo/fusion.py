import threading
import time
from typing import List, Any, Optional
from dataclasses import dataclass
from .core import Frame

@dataclass
class Observation:
    """Base class for a timestamped multimodal observation."""
    timestamp: float
    payload: Any

@dataclass
class VideoObservation(Observation):
    """A visual observation (e.g., a CUDA Frame)."""
    payload: Frame

@dataclass
class AudioObservation(Observation):
    """An audio observation (e.g., a Transcript or AudioChunk)."""
    duration: float

@dataclass
class MultimodalContext:
    """A temporal grouping of video and audio observations."""
    reference_timestamp: float
    video: List[VideoObservation]
    audio: List[AudioObservation]


class TemporalFusion:
    """
    A temporal synchronization buffer for aligning independent video and audio observations.
    Does not perform semantic reasoning.
    """
    def __init__(self, max_history_s: float = 30.0):
        """
        Args:
            max_history_s: Maximum age of observations to retain before garbage collection.
                           This bounds memory usage, particularly important for GPU Frames.
        """
        self.max_history_s = max_history_s
        self._lock = threading.Lock()
        
        self._video_buffer: List[VideoObservation] = []
        self._audio_buffer: List[AudioObservation] = []

        # Telemetry & Observability
        self._queries_count = 0
        self._latest_skew_ms = 0.0
        self._total_skew_ms = 0.0
        self._max_skew_ms = 0.0
        self._latest_lookup_latency_ms = 0.0
        self._total_lookup_latency_ms = 0.0

    def _evict_stale(self, latest_ts: float):
        """Remove observations that are older than the maximum history window."""
        threshold = latest_ts - self.max_history_s
        
        # Keep observations that overlap with the threshold or are newer
        self._video_buffer = [v for v in self._video_buffer if v.timestamp >= threshold]
        # For audio, keep if it ends after the threshold
        self._audio_buffer = [a for a in self._audio_buffer if (a.timestamp + a.duration) >= threshold]

    def add_video(self, frame: Frame, timestamp: float):
        """
        Add a visual observation to the fusion buffer.
        """
        obs = VideoObservation(timestamp=timestamp, payload=frame)
        with self._lock:
            self._video_buffer.append(obs)
            self._video_buffer.sort(key=lambda x: x.timestamp)
            self._evict_stale(timestamp)

    def add_audio(self, payload: Any, timestamp: float, duration: float):
        """
        Add an audio observation to the fusion buffer.
        """
        obs = AudioObservation(timestamp=timestamp, payload=payload, duration=duration)
        with self._lock:
            self._audio_buffer.append(obs)
            self._audio_buffer.sort(key=lambda x: x.timestamp)
            self._evict_stale(timestamp)

    def context(self, timestamp: float, window: float = 1.5) -> MultimodalContext:
        """
        Retrieve a MultimodalContext around a specific timestamp.
        
        Args:
            timestamp: The reference timestamp to align around.
            window: The +/- tolerance in seconds to include observations.
            
        Returns:
            A MultimodalContext containing matched observations.
        """
        t0 = time.time()
        min_ts = timestamp - window
        max_ts = timestamp + window
        
        with self._lock:
            matched_video = [
                v for v in self._video_buffer
                if min_ts <= v.timestamp <= max_ts
            ]
            
            # For audio, check if the duration interval overlaps with the window
            matched_audio = [
                a for a in self._audio_buffer
                if a.timestamp <= max_ts and (a.timestamp + a.duration) >= min_ts
            ]
            
            # Skew calculation: difference between matched video and audio timestamps
            skew_ms = 0.0
            if matched_video and matched_audio:
                # Discrepancy between closest or latest matched video and audio timestamps
                v_ts = matched_video[-1].timestamp
                a_ts = matched_audio[-1].timestamp
                skew_ms = abs(v_ts - a_ts) * 1000.0
            elif matched_video:
                skew_ms = abs(matched_video[-1].timestamp - timestamp) * 1000.0
            elif matched_audio:
                skew_ms = abs(matched_audio[-1].timestamp - timestamp) * 1000.0

            t1 = time.time()
            lookup_latency_ms = (t1 - t0) * 1000.0

            self._queries_count += 1
            self._latest_skew_ms = skew_ms
            self._total_skew_ms += skew_ms
            if skew_ms > self._max_skew_ms:
                self._max_skew_ms = skew_ms

            self._latest_lookup_latency_ms = lookup_latency_ms
            self._total_lookup_latency_ms += lookup_latency_ms
            
        return MultimodalContext(
            reference_timestamp=timestamp,
            video=matched_video,
            audio=matched_audio
        )

    def latest_context(self, window: float = 1.5) -> Optional[MultimodalContext]:
        """
        Retrieve a MultimodalContext around the most recent observation's timestamp.
        """
        with self._lock:
            latest_ts = 0.0
            if self._video_buffer:
                latest_ts = max(latest_ts, self._video_buffer[-1].timestamp)
            if self._audio_buffer:
                latest_ts = max(latest_ts, self._audio_buffer[-1].timestamp + self._audio_buffer[-1].duration)
                
            if latest_ts == 0.0:
                return None
                
        return self.context(latest_ts, window=window)

    def stats(self) -> dict:
        """Return operational and synchronization metrics for TemporalFusion."""
        with self._lock:
            avg_skew = self._total_skew_ms / max(1, self._queries_count)
            avg_lookup = self._total_lookup_latency_ms / max(1, self._queries_count)
            return {
                "video_buffer_size": len(self._video_buffer),
                "audio_buffer_size": len(self._audio_buffer),
                "queries_count": self._queries_count,
                "latest_skew_ms": round(self._latest_skew_ms, 3),
                "avg_skew_ms": round(avg_skew, 3),
                "max_skew_ms": round(self._max_skew_ms, 3),
                "latest_lookup_latency_ms": round(self._latest_lookup_latency_ms, 3),
                "avg_lookup_latency_ms": round(avg_lookup, 3),
            }
