# Velo Native Audio Ingestion (V1.8)

## 1. Overview

Velo V1.8 introduces native WebRTC / LiveKit audio ingestion. The native Rust transport directly ingests Opus-encoded audio tracks, decodes them to 16-bit linear PCM (`i16`), and passes them as normalized `AudioChunk` objects to Python via zero-copy tensor wrapping.

```
                    LiveKit SFU / WebRTC
                             │
            ┌────────────────┴────────────────┐
            │ [RTP H.264 @ 90kHz]             │ [RTP Opus @ 48kHz]
            ▼                                 ▼
      Native NVDEC                      Native Opus C-API
            │ (CUDA Frame)                    │ (PCM i16 Bytes)
            ▼                                 ▼
       Stream.next()                  Stream.next_audio()
            │                                 │
            ▼                                 ▼
       AIScheduler                      AudioScheduler
            │                                 │
            ▼                                 ▼
        VLM / Agent                       VAD / ASR
            │                                 │
            └────────────────┬────────────────┘
                             │
                             ▼
                      TemporalFusion
```

---

## 2. Public API

### 2.1 `Stream.next_audio()`
`Stream.next_audio()` retrieves the next native audio chunk from the WebRTC peer connection.

```python
import velo

stream, sdp_answer = velo.connect(sdp_offer)

# Ingest video frame
frame = stream.next()
video_tensor = frame.to_torch()  # GPU CUDA tensor

# Ingest audio chunk
chunk = stream.next_audio()
print(chunk.sample_rate)  # 48000
print(chunk.channels)     # 2 (stereo)
print(chunk.timestamp)    # Monotonic seconds since session start
print(chunk.duration)     # e.g., 0.02 (20ms)
print(chunk.samples)      # PyTorch Float32 Tensor normalized in [-1.0, 1.0]

stream.close()
```

### 2.2 Connecting to AudioScheduler
```python
from velo.audio_scheduler import AudioScheduler

# Configure scheduler for native 48kHz stereo stream
audio_scheduler = AudioScheduler(
    chunk_duration_s=0.5,
    sample_rate=48000,
    channels=2,
    max_buffered_s=30.0
)

# Submit native chunks
audio_scheduler.submit(chunk)

# Acquire consolidated 500ms chunk for ASR / VAD
speech_chunk = audio_scheduler.acquire()
```

---

## 3. Timestamp Semantics & Multimodal Synchronization

Video (H.264 @ 90 kHz) and Audio (Opus @ 48 kHz) RTP packets arrive with independent 32-bit RTP clocks.

Velo translates both media clocks to a unified, monotonic wall-clock domain (seconds since stream start $t_0$):
$$\text{timestamp}_{\text{audio}} = t_{audio, 0} + \frac{RTP_{audio} \ominus RTP_{audio, 0}}{48000.0}$$
$$\text{timestamp}_{\text{video}} = t_{video, 0} + \frac{RTP_{video} \ominus RTP_{video, 0}}{90000.0}$$

This allows `TemporalFusion` to accurately correlate video frames and audio observations into aligned multimodal contexts:

```python
from velo.fusion import TemporalFusion

fusion = TemporalFusion(max_history_s=10.0)

# Add synchronized observations
fusion.add_video(frame, timestamp=frame.timestamp)
fusion.add_audio(transcript, timestamp=transcript.start_timestamp, duration=speech_chunk.duration)

# Query multimodal context around T = 12.5s within +/- 0.5s window
context = fusion.context(timestamp=12.5, window=0.5)
```

---

## 4. Native Threading & Memory Architecture

- **Dedicated Tokio Worker:** All WebRTC RTP reception runs on a dedicated background worker thread.
- **Lockless Queues & Drop-Oldest Backpressure:**
  - Video queue capacity: 3 frames (drop oldest).
  - Audio queue capacity: 100 chunks (~2 seconds of 20ms audio, drop oldest).
  - Audio buffer memory is strictly bounded (< 400 KB).
- **Zero-Copy Tensor Construction:** PCM byte slices crossing into Python are converted via `torch.frombuffer(raw_bytes, dtype=torch.int16).to(torch.float32) / 32768.0`.
- **GIL Management:** Tokio async tasks acquire the GIL only briefly when constructing `PyBytes` objects, and `Stream.next_audio()` releases the GIL (`py.allow_threads`) while waiting for incoming audio.
- **Graceful Shutdown:** `stream.close()` unblocks both video and audio worker loops, joins native threads with GIL released, and drains queues under the GIL.
