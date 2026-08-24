# Velo Performance Benchmarks

Performance metrics captured under deterministic H.264 RTP ingestion on Windows 11.

---

## 1. Verified Performance Baseline (Target 30 FPS)

The following metrics represent a 15-second sustained targeted 30 FPS stream:

| Metric | Measured Value | Description |
|---|---|---|
| **Stream Duration** | 15.01 s | Ingestion test window |
| **Total Decoded Frames** | 448 | Number of H.264 frames processed |
| **Ingest Rate** | 29.85 FPS | Target frame rate of input stream |
| **GIL next() Wait** | 32.748 ms | Time spent waiting inside next() releasing the GIL |
| **PyTorch Tensor mapping** | 0.738 ms | DLPack mapping and synchronization time |
| **CPU Utilization** | 6.4 % | Process CPU usage (extremely low due to hardware decode) |
| **PyTorch VRAM** | 0.0 MB | Allocated PyTorch tensor memory on `cuda:0` |
| **Queue Frame Drops** | 0 | Bounded buffer drops |
| **NVDEC Decode Errors** | 0 | Hardware decoding errors |

---

## 2. Unthrottled Hardware Ceiling

The following metrics represent Velo's absolute decoding throughput limit under unthrottled ingestion:

| Metric | Measured Value | Description |
|---|---|---|
| **Maximum Decoded Rate** | **911.11 FPS** | Absolute hardware decoding ceiling |
| **PyTorch Tensor mapping** | 0.465 ms | DLPack mapping under full load |
| **Queue Drops** | 3 | Drops under max-throughput backpressure |

---

## 3. Architectural Comparison: Velo vs. aiortc

Below is an architectural analysis comparing the zero-copy Velo pipeline to the standard Python CPU-decoding pipeline (`aiortc` + `PyAV` / `FFmpeg`):

### Velo Pipeline
- **Decode Location**: GPU (NVDEC hardware)
- **Memory Location**: GPU VRAM directly
- **Python GIL Contention**: **Extremely Low** (released during WebRTC network waiting and native decoding)
- **Host-to-Device Copy**: **Zero** (frames never land in host RAM)
- **CPU Overhead**: **Very Low** (only network packetization and signaling)

### CPU Pipeline (aiortc / PyAV)
- **Decode Location**: CPU (software H.264 decoder)
- **Memory Location**: Host CPU RAM (numpy arrays)
- **Python GIL Contention**: **High** (software decoding runs inside/alongside the interpreter)
- **Host-to-Device Copy**: **Required** (every frame must be copied from CPU RAM to GPU VRAM for PyTorch)
- **CPU Overhead**: **Very High** (software decoding scales poorly with resolution and stream count)
