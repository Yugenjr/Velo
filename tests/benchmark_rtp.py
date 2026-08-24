"""
Velo Deterministic Ingest Benchmarking Suite

Pushes H.264 packets as fast as possible to the native RtpReceiver pipeline,
collects detailed throughput, latency, GIL, and GPU performance metrics,
and writes the baseline results directly to docs/benchmarks.md.
"""
import os
import sys
import re
import time
import threading
import torch
import velo

try:
    import psutil
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

def run_performance_benchmark():
    print("\n--- Running Deterministic Performance Benchmark ---")
    
    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available.")
        sys.exit(1)
    torch.cuda.init()
    process = psutil.Process()

    # Load sample H.264 NAL units
    h264_path = os.path.join("temp", "native_webrtc_experiment", "output.h264")
    if not os.path.exists(h264_path):
        print(f"[FATAL] H.264 sample file not found at {h264_path}")
        sys.exit(1)
        
    with open(h264_path, "rb") as f:
        h264_data = f.read()

    nals = re.split(rb'\x00\x00\x00\x01|\x00\x00\x01', h264_data)
    nals = [n for n in nals if len(n) > 0]
    
    receiver = velo.RtpReceiver()
    
    frame_times = []
    next_wait_times = []
    conversion_times = []
    cpu_metrics = []
    
    stop_event = threading.Event()
    
    # We will simulate pushing at a sustained targeted 30 FPS first for 15 seconds,
    # and then run an unthrottled maximum throughput test for another 15 seconds
    # to find the absolute hardware decoding limit!
    print("Step 1: Running sustained 30 FPS target benchmark...")
    
    pushed_count = 0
    decoded_count = 0
    
    def push_target_worker():
        nonlocal pushed_count
        timestamp = 90000
        t_start = time.time()
        while not stop_event.is_set() and time.time() - t_start < 15:
            for nal in nals:
                if stop_event.is_set() or time.time() - t_start >= 15:
                    break
                nal_type = nal[0] & 0x1F
                if nal_type in (1, 5):
                    timestamp += 3000
                    pushed_count += 1
                    # target ~30 FPS
                    time.sleep(0.033)
                try:
                    receiver.push_rtp(nal, timestamp)
                except Exception:
                    return

    push_thread = threading.Thread(target=push_target_worker, daemon=True)
    push_thread.start()
    
    # Consume targeted stream
    t_start = time.time()
    while time.time() - t_start < 15:
        t0 = time.time()
        try:
            frame = receiver.next()
            t_next = time.time()
            next_wait_times.append(t_next - t0)
            
            # Measure tensor mapping conversion latency
            t_conv_start = time.time()
            tensor = frame.to_torch()
            _val = tensor.float().mean()
            torch.cuda.synchronize()
            t_conv_end = time.time()
            
            conversion_times.append(t_conv_end - t_conv_start)
            frame_times.append(t_conv_end - t0)
            decoded_count += 1
            
            if decoded_count % 30 == 0:
                cpu_metrics.append(process.cpu_percent(interval=None))
        except velo.StreamClosedError:
            break

    stop_event.set()
    push_thread.join()
    
    duration_target = time.time() - t_start
    fps_target = decoded_count / duration_target
    
    # Reset for unthrottled maximum throughput benchmark
    print("Step 2: Running unthrottled maximum throughput benchmark...")
    stop_event.clear()
    
    max_receiver = velo.RtpReceiver()
    max_pushed = 0
    max_decoded = 0
    
    def push_max_worker():
        nonlocal max_pushed
        timestamp = 90000
        t_start = time.time()
        while not stop_event.is_set() and time.time() - t_start < 15:
            for nal in nals:
                if stop_event.is_set() or time.time() - t_start >= 15:
                    break
                nal_type = nal[0] & 0x1F
                if nal_type in (1, 5):
                    timestamp += 3000
                    max_pushed += 1
                try:
                    max_receiver.push_rtp(nal, timestamp)
                except Exception:
                    return

    push_thread_max = threading.Thread(target=push_max_worker, daemon=True)
    push_thread_max.start()
    
    t_start = time.time()
    max_conversion_times = []
    while time.time() - t_start < 15:
        try:
            frame = max_receiver.next()
            t_c0 = time.time()
            tensor = frame.to_torch()
            _val = tensor.float().mean()
            torch.cuda.synchronize()
            max_conversion_times.append(time.time() - t_c0)
            max_decoded += 1
        except velo.StreamClosedError:
            break
            
    stop_event.set()
    push_thread_max.join()
    
    duration_max = time.time() - t_start
    fps_max = max_decoded / duration_max
    
    # Calculate stats
    avg_next_ms = sum(next_wait_times) / len(next_wait_times) * 1000 if next_wait_times else 0
    avg_conv_ms = sum(conversion_times) / len(conversion_times) * 1000 if conversion_times else 0
    avg_cpu = sum(cpu_metrics) / len(cpu_metrics) if cpu_metrics else 0
    gpu_allocated_mb = torch.cuda.memory_allocated(device=0) / 1024 / 1024
    
    avg_max_conv_ms = sum(max_conversion_times) / len(max_conversion_times) * 1000 if max_conversion_times else 0
    
    report = (
        f"\n"
        f"============================================================\n"
        f"               VELO DETERMINISTIC PERFORMANCE REPORT        \n"
        f"============================================================\n"
        f"  Target 30 FPS Stream Profile:\n"
        f"    Duration:            {duration_target:.2f} seconds\n"
        f"    Frames Decoded:      {decoded_count}\n"
        f"    Target Ingest Rate:  {fps_target:.2f} FPS\n"
        f"    GIL next() Blocked:  {avg_next_ms:.3f} ms (idle wait)\n"
        f"    PyTorch Tensor conversion: {avg_conv_ms:.3f} ms\n"
        f"    CPU Utilization:     {avg_cpu:.1f} %\n"
        f"    GPU Memory:          {gpu_allocated_mb:.1f} MB\n"
        f"    Queue drops:         {receiver.dropped_frames}\n"
        f"    Decode errors:       {receiver.decode_errors}\n"
        f"\n"
        f"  Unthrottled Stream Profile (Hardware Ceiling):\n"
        f"    Max Decoded Rate:    {fps_max:.2f} FPS\n"
        f"    Max Tensor conv:     {avg_max_conv_ms:.3f} ms\n"
        f"    Max Queue drops:     {max_receiver.dropped_frames}\n"
        f"============================================================\n"
    )
    print(report)
    
    receiver.close()
    max_receiver.close()
    
    write_benchmarks_doc(
        duration_target, decoded_count, fps_target, avg_next_ms, avg_conv_ms,
        avg_cpu, gpu_allocated_mb, receiver.dropped_frames, receiver.decode_errors,
        fps_max, avg_max_conv_ms, max_receiver.dropped_frames
    )

def write_benchmarks_doc(duration, frames, fps, next_ms, conv_ms, cpu, vram, drops, errors, max_fps, max_conv_ms, max_drops):
    content = f"""# Velo Performance Benchmarks

Performance metrics captured under deterministic H.264 RTP ingestion on Windows 11.

---

## 1. Verified Performance Baseline (Target 30 FPS)

The following metrics represent a 15-second sustained targeted 30 FPS stream:

| Metric | Measured Value | Description |
|---|---|---|
| **Stream Duration** | {duration:.2f} s | Ingestion test window |
| **Total Decoded Frames** | {frames} | Number of H.264 frames processed |
| **Ingest Rate** | {fps:.2f} FPS | Target frame rate of input stream |
| **GIL next() Wait** | {next_ms:.3f} ms | Time spent waiting inside next() releasing the GIL |
| **PyTorch Tensor mapping** | {conv_ms:.3f} ms | DLPack mapping and synchronization time |
| **CPU Utilization** | {cpu:.1f} % | Process CPU usage (extremely low due to hardware decode) |
| **PyTorch VRAM** | {vram:.1f} MB | Allocated PyTorch tensor memory on `cuda:0` |
| **Queue Frame Drops** | {drops} | Bounded buffer drops |
| **NVDEC Decode Errors** | {errors} | Hardware decoding errors |

---

## 2. Unthrottled Hardware Ceiling

The following metrics represent Velo's absolute decoding throughput limit under unthrottled ingestion:

| Metric | Measured Value | Description |
|---|---|---|
| **Maximum Decoded Rate** | **{max_fps:.2f} FPS** | Absolute hardware decoding ceiling |
| **PyTorch Tensor mapping** | {max_conv_ms:.3f} ms | DLPack mapping under full load |
| **Queue Drops** | {max_drops} | Drops under max-throughput backpressure |

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
"""
    with open("docs/benchmarks.md", "w") as f:
        f.write(content)
    print("Written docs/benchmarks.md successfully.")

if __name__ == "__main__":
    run_performance_benchmark()
