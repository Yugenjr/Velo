"""
Velo V0.8 Benchmarking Suite

Runs a WebRTC receiver server, collects performance metrics under active stream
for 30 seconds, and prints a detailed execution profile.
"""
import os
import sys
import time
import asyncio
import threading
from aiohttp import web
import velo
import torch

try:
    import psutil
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

# Initialize PyTorch CUDA
if not torch.cuda.is_available():
    print("[FATAL] CUDA not available.")
    sys.exit(1)
torch.cuda.init()
process = psutil.Process()

# Benchmark stats
frame_times = []
next_wait_times = []
processing_times = []
cpu_percentages = []
dropped_counts = []
decode_errors = []

stream = None
benchmark_thread = None
benchmark_running = False

def run_benchmark():
    global stream, benchmark_running
    print("[Benchmark] Starting 30-second performance baseline measurement...")
    
    frames_count = 0
    t_start = time.time()
    benchmark_running = True
    
    try:
        # Measure baseline CPU
        process.cpu_percent(interval=None)
        
        while time.time() - t_start < 30:
            t0 = time.time()
            
            # 1. Measure wait time in stream.next() (GIL release overhead)
            frame = stream.next()
            t_next = time.time()
            next_wait_times.append(t_next - t0)
            
            # 2. Simulate model processing work (e.g. PyTorch mean + synchronization)
            t_proc_start = time.time()
            tensor = frame.to_torch()
            # Perform dummy CUDA ops to ensure GPU work is submitted
            dummy = tensor.float().mean()
            torch.cuda.synchronize()
            t_proc_end = time.time()
            
            processing_times.append(t_proc_end - t_proc_start)
            frame_times.append(t_proc_end - t0)
            
            frames_count += 1
            
            # Capture CPU and queue stats periodically
            if frames_count % 30 == 0:
                cpu_percentages.append(process.cpu_percent(interval=None))
                dropped_counts.append(stream.dropped_frames)
                decode_errors.append(stream.decode_errors)
                
    except velo.StreamClosedError:
        print("[Benchmark] Stream closed before benchmark finished.")
    except Exception as e:
        print(f"[Benchmark] Error: {e}")
    finally:
        benchmark_running = False
        duration = time.time() - t_start
        print(f"[Benchmark] Completed. Duration: {duration:.2f}s, Total Frames: {frames_count}")
        print_results(frames_count, duration)

def print_results(frames, duration):
    if not frame_times:
        print("No benchmark data captured.")
        return
        
    avg_fps = frames / duration if duration > 0 else 0
    avg_next = sum(next_wait_times) / len(next_wait_times) * 1000
    avg_proc = sum(processing_times) / len(processing_times) * 1000
    avg_cpu = sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0
    final_drops = dropped_counts[-1] if dropped_counts else 0
    final_errors = decode_errors[-1] if decode_errors else 0
    
    # Query GPU memory allocated by PyTorch (excludes PyNvVideoCodec direct VRAM)
    gpu_allocated_mb = torch.cuda.memory_allocated(device=0) / 1024 / 1024
    
    report = (
        f"\n"
        f"============================================================\n"
        f"                   VELO V0.8 BENCHMARK REPORT               \n"
        f"============================================================\n"
        f"  Total Duration:      {duration:.2f} seconds\n"
        f"  Total Frames:        {frames}\n"
        f"  Average Throughput:  {avg_fps:.2f} FPS\n"
        f"  GIL/next() Wait:     {avg_next:.3f} ms (average)\n"
        f"  PyTorch CUDA Ops:    {avg_proc:.3f} ms (average)\n"
        f"  CPU Utilization:     {avg_cpu:.1f} % (process average)\n"
        f"  PyTorch VRAM:        {gpu_allocated_mb:.1f} MB\n"
        f"  Total Queue Drops:   {final_drops}\n"
        f"  Total Decode Errors: {final_errors}\n"
        f"============================================================\n"
    )
    print(report)
    
    # Write directly to docs/benchmarks.md
    write_benchmark_doc(duration, frames, avg_fps, avg_next, avg_proc, avg_cpu, gpu_allocated_mb, final_drops, final_errors)

def write_benchmark_doc(duration, frames, fps, next_ms, proc_ms, cpu, vram, drops, errors):
    content = f"""# Velo Performance Benchmarks

Performance metrics captured under loopback WebRTC camera streaming on Windows.

---

## 1. Verified Performance Baseline (V0.8)

The following metrics represent a 30-second sustained execution benchmark:

| Metric | Measured Value | Description |
|---|---|---|
| **Stream Duration** | {duration:.2f} s | Test window duration |
| **Total Decoded Frames** | {frames} | Number of WebRTC H.264 frames processed |
| **Average Throughput** | {fps:.2f} FPS | Processing rate (limited by camera capture rate) |
| **GIL/next() Wait** | {next_ms:.3f} ms | Average time blocked in `stream.next()` releasing the GIL |
| **PyTorch CUDA Ops** | {proc_ms:.3f} ms | Average time to convert DLPack and execute mean() on GPU |
| **CPU Utilization** | {cpu:.1f} % | Process CPU usage (very low due to hardware decode) |
| **PyTorch VRAM** | {vram:.1f} MB | Allocated PyTorch tensor memory on `cuda:0` |
| **Queue Frame Drops** | {drops} | Frames dropped due to bounded buffer capacity (size=3) |
| **NVDEC Decode Errors** | {errors} | Hardware decoding errors |

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
"""
    # Save the document
    with open("docs/benchmarks.md", "w") as f:
        f.write(content)
    print("[Benchmark] Written docs/benchmarks.md successfully.")

async def index(request):
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "index.html")
    html = open(path, "r").read()
    return web.Response(content_type="text/html", text=html)

async def offer(request):
    global stream, benchmark_thread
    params = await request.json()
    sdp = params["sdp"]
    
    print("[Server] Received WebRTC offer. Connecting Velo stream...")
    stream, answer_sdp = velo.connect(sdp)
    
    benchmark_thread = threading.Thread(target=run_benchmark, daemon=True)
    benchmark_thread.start()
    
    return web.json_response({
        "sdp": answer_sdp,
        "type": "answer"
    })

async def shutdown(request):
    global stream
    if stream is not None:
        stream.close()
        stream = None
    return web.Response(text="closed")

if __name__ == "__main__":
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_post("/shutdown", shutdown)
    
    print("=" * 60)
    # Serves the exact same index.html from examples, but runs the benchmark profile.
    print("  Velo V0.8 Performance Benchmark Server")
    print("  1. Open http://localhost:8080 in your browser")
    print("  2. Click 'Start Streaming to Velo'")
    print("  3. Benchmark will run automatically for 30 seconds")
    print("=" * 60)
    web.run_app(app, host="0.0.0.0", port=8080)
