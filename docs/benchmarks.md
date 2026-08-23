# Velo Performance Benchmarks

Performance metrics captured under loopback WebRTC camera streaming on Windows.

---

## 1. Verified Performance Baseline (V0.8)

The following metrics represent a 30-second sustained execution benchmark:

| Metric | Measured Value | Description |
|---|---|---|
| **Stream Duration** | 30.09 s | Test window duration |
| **Total Decoded Frames** | 219 | Number of WebRTC H.264 frames processed |
| **Average Throughput** | 7.28 FPS | Processing rate (limited by camera capture rate) |
| **GIL/next() Wait** | 133.568 ms | Average time blocked in `stream.next()` releasing the GIL |
| **PyTorch CUDA Ops** | 3.772 ms | Average time to convert DLPack and execute mean() on GPU |
| **CPU Utilization** | 8.6 % | Process CPU usage (very low due to hardware decode) |
| **PyTorch VRAM** | 0.0 MB | Allocated PyTorch tensor memory on `cuda:0` |
| **Queue Frame Drops** | 0 | Frames dropped due to bounded buffer capacity (size=3) |
| **NVDEC Decode Errors** | 0 | Hardware decoding errors |

---

## 2. Comparison: Velo vs. CPU-Decoded Pipeline (aiortc)

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
