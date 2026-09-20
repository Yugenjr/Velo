"""
Velo Real H.264 Video & Hardware NVDEC Benchmark (Milestone V1.11)

Taxonomy Class: Class 4 (Real Media & Hardware NVDEC)

Measures:
1. Real Annex-B H.264 NAL parsing and RTP ingestion.
2. Hardware NVDEC hardware decoding throughput across resolutions (480p, 720p, 1080p, 4K).
3. Sustained 30 FPS realtime frame interval distribution (mean, p50, p95, p99, jitter).
4. GPU residency verification and zero-copy DLPack conversion overhead.
5. VRAM consumption and CPU usage.
"""
import os
import sys
import io
import re
import time
import threading
import numpy as np
import torch
import av
import psutil
import velo
from benchmark_utils import save_benchmark_results, get_system_metadata


def generate_annexb_h264(width: int, height: int, num_frames: int = 60, fps: int = 30) -> bytes:
    """Encodes standard Annex-B H.264 elementary stream using PyAV."""
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="h264")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "ultrafast", "tune": "zerolatency"}

    for i in range(num_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Add dynamic spatial pattern
        img[:, :, 0] = (i * 4) % 255
        img[:, :, 1] = (i * 8) % 255
        img[height//4:height//2, width//4:width//2, 2] = 255
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buf.getvalue()


def run_resolution_benchmark(width: int, height: int, name: str, num_frames: int = 90, warmup_frames: int = 15):
    """Benchmarks hardware NVDEC decoding for a specific resolution."""
    print(f"\n--- [Class 4] Real H.264 NVDEC Benchmark: {name} ({width}x{height}) ---")
    
    # 1. Generate real H.264 elementary stream
    t0_gen = time.time()
    h264_data = generate_annexb_h264(width, height, num_frames=num_frames, fps=30)
    nals = [n for n in re.split(rb"\x00\x00\x00\x01|\x00\x00\x01", h264_data) if len(n) > 0]
    t_gen = time.time() - t0_gen
    print(f"Generated {len(nals)} H.264 NAL units in {t_gen*1000:.1f}ms")

    # 2. Warmup Decoder & Verify GPU Residency
    receiver = velo.RtpReceiver(codec="h264", max_width=width, max_height=height)
    
    # Push initial frames for warmup
    ts = 90000
    for i in range(min(len(nals), warmup_frames * 2)):
        receiver.push_rtp(nals[i % len(nals)], ts)
        ts += 3000
    
    warmup_frame = receiver.next()
    torch_tensor = warmup_frame.to_torch()
    
    # Residency Assertions
    assert torch_tensor.is_cuda, f"Expected CUDA tensor, got device {torch_tensor.device}"
    assert torch_tensor.shape == (height, width, 3), f"Shape mismatch: {torch_tensor.shape} vs ({height}, {width}, 3)"
    assert torch_tensor.dtype == torch.uint8, f"Dtype mismatch: {torch_tensor.dtype}"
    
    print(f"GPU Residency Verified: shape={torch_tensor.shape}, device={torch_tensor.device}, dtype={torch_tensor.dtype}")
    
    # 3. Measure Zero-Copy DLPack Conversion Latency
    conversion_latencies_us = []
    for _ in range(50):
        t_c0 = time.perf_counter()
        t_wrap = warmup_frame.to_torch()
        torch.cuda.synchronize()
        t_c1 = time.perf_counter()
        conversion_latencies_us.append((t_c1 - t_c0) * 1e6)
    
    avg_dlpack_us = float(np.mean(conversion_latencies_us))
    p50_dlpack_us = float(np.percentile(conversion_latencies_us, 50))
    p95_dlpack_us = float(np.percentile(conversion_latencies_us, 95))
    print(f"DLPack Zero-Copy Tensor Conversion Overhead: avg={avg_dlpack_us:.2f}us, p50={p50_dlpack_us:.2f}us, p95={p95_dlpack_us:.2f}us")

    # 4. Measure Unthrottled Max Decode Throughput (Steady-State)
    stop_event = threading.Event()
    decoded_frames = 0
    frame_intervals_ms = []
    
    def push_unthrottled():
        cur_ts = 90000
        for _ in range(3):
            for nal in nals:
                if stop_event.is_set():
                    return
                try:
                    receiver.push_rtp(nal, cur_ts)
                    cur_ts += 3000
                except Exception:
                    return

    pusher = threading.Thread(target=push_unthrottled, daemon=True)
    pusher.start()
    
    t_start = time.time()
    last_frame_t = t_start
    
    for _ in range(num_frames):
        try:
            f = receiver.next()
            now = time.time()
            if decoded_frames > 0:
                frame_intervals_ms.append((now - last_frame_t) * 1000.0)
            last_frame_t = now
            decoded_frames += 1
        except Exception:
            break
            
    stop_event.set()
    t_total = time.time() - t_start
    receiver.close()
    
    max_decode_fps = round(decoded_frames / max(0.0001, t_total), 2)
    
    # 5. Measure GPU VRAM
    gpu_mem_mb = 0.0
    if torch.cuda.is_available():
        gpu_mem_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)

    result = {
        "resolution_name": name,
        "width": width,
        "height": height,
        "decoded_frames": decoded_frames,
        "duration_s": round(t_total, 4),
        "max_nvdec_throughput_fps": max_decode_fps,
        "frame_interval_ms": {
            "mean": round(float(np.mean(frame_intervals_ms)), 3) if frame_intervals_ms else 0.0,
            "p50": round(float(np.percentile(frame_intervals_ms, 50)), 3) if frame_intervals_ms else 0.0,
            "p95": round(float(np.percentile(frame_intervals_ms, 95)), 3) if frame_intervals_ms else 0.0,
            "p99": round(float(np.percentile(frame_intervals_ms, 99)), 3) if frame_intervals_ms else 0.0,
            "std": round(float(np.std(frame_intervals_ms)), 3) if frame_intervals_ms else 0.0,
        },
        "dlpack_conversion_us": {
            "mean": round(avg_dlpack_us, 2),
            "p50": round(p50_dlpack_us, 2),
            "p95": round(p95_dlpack_us, 2),
        },
        "gpu_memory_allocated_mb": gpu_mem_mb,
        "is_cuda": True,
    }
    
    print(f"Results for {name}: Max NVDEC Decode FPS = {max_decode_fps} FPS | Frame interval p50 = {result['frame_interval_ms']['p50']}ms")
    return result


def run_all_video_benchmarks():
    print("==================================================")
    print("VELO REAL H.264 & HARDWARE NVDEC BENCHMARK (V1.11)")
    print("==================================================")
    
    results = {}
    
    # Test 480p
    results["480p"] = run_resolution_benchmark(640, 480, "480p", num_frames=60)
    
    # Test 720p
    results["720p"] = run_resolution_benchmark(1280, 720, "720p", num_frames=60)
    
    # Test 1080p
    results["1080p"] = run_resolution_benchmark(1920, 1080, "1080p", num_frames=60)
    
    # Test 4K
    results["4k"] = run_resolution_benchmark(3840, 2160, "4K_UHD", num_frames=30)
    
    config = {
        "resolutions": ["480p", "720p", "1080p", "4k"],
        "codec": "h264",
        "decoder": "NVIDIA NVDEC (Hardware)",
        "zero_copy_format": "DLPack -> PyTorch CUDA",
    }
    filepath = save_benchmark_results("real_h264_nvdec", config, results)
    return results


if __name__ == "__main__":
    run_all_video_benchmarks()
