import threading
import torch
import time
from collections import deque
from typing import Optional, List, Dict
from .audio import AudioChunk
from .scheduler import SchedulerClosedError

class AudioScheduler:
    """
    A synchronized scheduler for buffering and grouping audio chunks.
    
    Accepts arbitrary-sized incoming AudioChunks and yields consistently sized
    AudioChunks (e.g., 500ms) for VAD/ASR processing.
    """
    def __init__(
        self,
        chunk_duration_s: float = 0.5,
        max_buffered_s: float = 30.0,
        sample_rate: int = 16000,
        channels: int = 1
    ):
        """
        Args:
            chunk_duration_s: The fixed duration in seconds to yield per acquire().
            max_buffered_s: The maximum duration of audio to buffer before dropping (backpressure).
            sample_rate: Expected sample rate of incoming audio.
            channels: Expected channels of incoming audio.
        """
        self.chunk_duration_s = chunk_duration_s
        self.max_buffered_s = max_buffered_s
        self.sample_rate = sample_rate
        self.channels = channels
        
        self.target_samples = int(self.chunk_duration_s * self.sample_rate * self.channels)
        
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._closed = False
        
        # Buffer of raw tensors
        self._buffer: deque = deque()
        self._buffered_samples = 0
        
        # We track the expected timestamp of the next yielded chunk based on the earliest seen audio
        self._current_timestamp: Optional[float] = None
        
        # Stats
        self._chunks_received = 0
        self._chunks_processed = 0
        self._chunks_dropped = 0
        self._total_latency = 0.0
        
    def submit(self, chunk: AudioChunk):
        """
        Submit a new AudioChunk into the scheduler.
        """
        if chunk.sample_rate != self.sample_rate or chunk.channels != self.channels:
            raise ValueError(f"Expected {self.sample_rate}Hz/{self.channels}ch, got {chunk.sample_rate}Hz/{chunk.channels}ch")
            
        with self._lock:
            if self._closed:
                return
                
            self._chunks_received += 1
            
            # Update start timestamp if this is the first chunk or buffer is empty
            if self._current_timestamp is None:
                self._current_timestamp = chunk.timestamp
            
            samples = chunk.samples
            num_samples = samples.size(0)
            
            # Enforce backpressure (drop oldest audio)
            max_total_samples = int(self.max_buffered_s * self.sample_rate * self.channels)
            while self._buffered_samples + num_samples > max_total_samples:
                if not self._buffer:
                    break
                dropped_samples = self._buffer.popleft()
                dropped_len = dropped_samples.size(0)
                self._buffered_samples -= dropped_len
                self._chunks_dropped += 1
                # Advance the timestamp because we dropped the start of our buffered stream
                if self._current_timestamp is not None:
                    self._current_timestamp += (dropped_len / (self.sample_rate * self.channels))
            
            self._buffer.append(samples)
            self._buffered_samples += num_samples
            
            if self._buffered_samples >= self.target_samples:
                self._cond.notify_all()
                
    def acquire(self) -> AudioChunk:
        """
        Block until a full chunk (chunk_duration_s) of audio is available, then return it.
        """
        with self._cond:
            while self._buffered_samples < self.target_samples and not self._closed:
                self._cond.wait()
                
            if self._closed and self._buffered_samples < self.target_samples:
                raise SchedulerClosedError("Scheduler closed while waiting for audio.")
                
            # We have enough samples to build a chunk
            gathered_tensors = []
            gathered_samples = 0
            
            while gathered_samples < self.target_samples and self._buffer:
                needed = self.target_samples - gathered_samples
                head = self._buffer[0]
                head_len = head.size(0)
                
                if head_len <= needed:
                    # Take the whole tensor
                    gathered_tensors.append(self._buffer.popleft())
                    gathered_samples += head_len
                    self._buffered_samples -= head_len
                else:
                    # Take a slice, keep the rest
                    gathered_tensors.append(head[:needed])
                    self._buffer[0] = head[needed:]
                    gathered_samples += needed
                    self._buffered_samples -= needed
                    
            # Concat the slices
            if len(gathered_tensors) == 1:
                final_samples = gathered_tensors[0]
            else:
                final_samples = torch.cat(gathered_tensors, dim=0)
                
            out_timestamp = self._current_timestamp if self._current_timestamp is not None else 0.0
            
            # Advance timestamp for the next chunk
            if self._current_timestamp is not None:
                self._current_timestamp += self.chunk_duration_s
                
            self._chunks_processed += 1
            
            # We don't have a direct incoming packet timestamp mapped to this exact sliced output, 
            # so latency tracking is more synthetic, but we can compute it if needed.
            
            return AudioChunk(
                samples=final_samples,
                sample_rate=self.sample_rate,
                channels=self.channels,
                timestamp=out_timestamp,
                duration=self.chunk_duration_s
            )
            
    def close(self):
        """Shut down the scheduler and unblock waiters."""
        with self._lock:
            self._closed = True
            self._cond.notify_all()
            
    def stats(self) -> Dict[str, float]:
        """Return statistics about audio scheduling."""
        with self._lock:
            total_samples_per_sec = self.sample_rate * self.channels
            buffered_duration = self._buffered_samples / total_samples_per_sec if total_samples_per_sec > 0 else 0.0
            avg_latency = self._total_latency / max(1, self._chunks_processed)
            
            return {
                "chunks_received": self._chunks_received,
                "chunks_processed": self._chunks_processed,
                "chunks_dropped": self._chunks_dropped,
                "buffered_duration": buffered_duration,
                "average_chunk_latency": avg_latency
            }
