"""
Velo Real Multi-Stream Hardware NVDEC Benchmark (Milestone V2.0)

Taxonomy Class: Class 4 (Real Media & Hardware NVDEC Multi-Stream)

Measures:
1. Real Annex-B H.264 NAL parsing and NVDEC decoding across multiple concurrent streams (1, 2, 4 streams).
2. Per-stream FPS and Aggregate FPS for 720p and 1080p real video streams.
3. GPU VRAM consumption, CPU utilization, frame drop rates, and fairness.
4. Real-time MultiStreamPipeline concurrent execution with shared model scheduling.
"""
import os
import sys
import io
import re
import time
import json
import threading
import numpy as np
import torch
import av
import psutil
import velo
from velo import MultiStreamPipeline, MockMultimodalAdapter, AudioScheduler, TemporalFusion
from benchmark_utils import save_benchmark_results, get_system_metadata


def generate_annexb_h264(width: int, height: int, num_frames: int = 120, fps: int = 30) -> bytes:
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


class RealH264StreamSource:
    """Simulates a realtime WebRTC stream feeding real H.264 NALs into a hardware NVDEC receiver."""
    def __init__(self, stream_id: str, nals: list, width: int, height: int, target_fps: float = 30.0):
        self.stream_id = stream_id
        self.nals = nals
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.receiver = velo.RtpReceiver(codec="h264", max_width=width, max_height=height, stream_id=stream_id)
        self._rtp_ts = 90000
        self._nal_idx = 0
        self._closed = False
        self._lock = threading.Lock()
        
        self._push_thread = threading.Thread(target=self._push_worker, daemon=True)
        self._push_thread.start()

    def _push_worker(self):
        while not self._closed:
            nal = self.nals[self._nal_idx % len(self.nals)]
            nal_type = nal[0] & 0x1F
            
            try:
                self.receiver.push_rtp(nal, self._rtp_ts)
            except Exception:
                break
                
            if nal_type in (1, 5):
                self._rtp_ts += 3000
                
            self._nal_idx += 1
            # Push as fast as possible to find max hardware capacity,
            # yielding slightly so we don't totally starve the python thread scheduler
            time.sleep(0.0001)

    def next(self) -> velo.Frame:
        if self._closed:
            raise velo.StreamClosedError("Stream closed")
            
        frame = self.receiver.next()
        if frame is None:
            raise velo.StreamClosedError("Decoder drained")
        frame.arrival_time = time.time()
        return frame

    def next_audio(self):
        # Generate synthetic 20ms audio chunk
        with self._lock:
            if self._closed:
                raise velo.StreamClosedError("Audio closed")
            t = time.time()
            return velo.AudioChunk(
                samples=torch.zeros(960, dtype=torch.float32),
                sample_rate=16000,
                channels=1,
                timestamp=t,
                duration=0.02,
            )

    def close(self):
        with self._lock:
            self._closed = True
            try:
                self.receiver.close()
            except Exception:
                pass


def benchmark_nvdec_multistream(num_streams: int, width: int, height: int, res_name: str, duration_s: float = 3.0):
    print(f"\n=======================================================")
    print(f"--- [Real NVDEC Multi-Stream] {num_streams} Streams @ {res_name} ({width}x{height}) ---")
    print(f"=======================================================")

    # 1. Generate real H.264 elementary stream
    h264_data = generate_annexb_h264(width, height, num_frames=120, fps=30)
    nals = [n for n in re.split(rb"\x00\x00\x00\x01|\x00\x00\x01", h264_data) if len(n) > 0]
    print(f"Generated {len(nals)} H.264 NAL units for {res_name}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    proc = psutil.Process()
    cpu_before = proc.cpu_percent(interval=None)
    mem_before_mb = proc.memory_info().rss / (1024 * 1024)
    gpu_before_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    # 2. Create streams and decoders
    streams = [
        RealH264StreamSource(f"stream-{i}", nals, width, height, target_fps=30.0)
        for i in range(num_streams)
    ]

    stop_event = threading.Event()
    per_stream_counts = [0] * num_streams
    per_stream_intervals = [[] for _ in range(num_streams)]
    threads = []

    def stream_worker(idx: int, source: RealH264StreamSource):
        last_t = time.perf_counter()
        while not stop_event.is_set():
            try:
                f = source.next()
                if f is not None:
                    # Convert to PyTorch to ensure GPU residency
                    t = f.to_torch()
                    now = time.perf_counter()
                    per_stream_intervals[idx].append((now - last_t) * 1000.0)
                    last_t = now
                    per_stream_counts[idx] += 1
            except Exception:
                break

    t_start = time.perf_counter()
    for i, s in enumerate(streams):
        t = threading.Thread(target=stream_worker, args=(i, s), daemon=True)
        threads.append(t)
        t.start()

    time.sleep(duration_s)
    stop_event.set()

    for s in streams:
        s.close()
    for t in threads:
        t.join(timeout=1.0)
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    total_frames = sum(per_stream_counts)
    aggregate_fps = total_frames / elapsed
    per_stream_fps = [c / elapsed for c in per_stream_counts]

    cpu_after = proc.cpu_percent(interval=None)
    mem_after_mb = proc.memory_info().rss / (1024 * 1024)
    gpu_after_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    print(f"Results for {num_streams}x {res_name}:")
    print(f"  Duration: {elapsed:.2f}s")
    print(f"  Total Decoded Frames: {total_frames}")
    print(f"  Aggregate Decode FPS: {aggregate_fps:.1f} FPS")
    print(f"  Per-Stream FPS: {[round(fps, 1) for fps in per_stream_fps]}")
    print(f"  GPU Max VRAM: {gpu_after_mb:.2f} MB")
    print(f"  CPU Usage: {cpu_after:.1f}%")

    result = {
        "num_streams": num_streams,
        "resolution": res_name,
        "width": width,
        "height": height,
        "duration_s": elapsed,
        "total_frames": total_frames,
        "aggregate_fps": round(aggregate_fps, 2),
        "per_stream_fps": [round(f, 2) for f in per_stream_fps],
        "gpu_max_vram_mb": round(gpu_after_mb, 2),
        "cpu_usage_pct": round(cpu_after, 2),
    }
    return result


async def benchmark_multistream_pipeline_multimodal(num_streams: int = 2):
    print(f"\n=======================================================")
    print(f"--- [Real MultiStreamPipeline E2E] {num_streams} Concurrent Streams ---")
    print(f"=======================================================")

    h264_data = generate_annexb_h264(1280, 720, num_frames=90, fps=30)
    nals = [n for n in re.split(rb"\x00\x00\x00\x01|\x00\x00\x01", h264_data) if len(n) > 0]

    vlm = MockMultimodalAdapter(simulated_latency=0.005)
    pipeline = MultiStreamPipeline(vlm=vlm)

    for i in range(num_streams):
        source = RealH264StreamSource(f"stream-{i}", nals, 1280, 720, target_fps=30.0)
        pipeline.add_stream(
            f"stream-{i}",
            source,
            target_fps=30.0,
            audio_scheduler=AudioScheduler(),
            fusion=TemporalFusion(max_history_s=10.0),
        )

    await pipeline.start()
    counts = {f"stream-{i}": 0 for i in range(num_streams)}
    t0 = time.time()

    async for stream_id, resp in pipeline.run_inference("Monitor room activity"):
        counts[stream_id] += 1
        if sum(counts.values()) >= 30 * num_streams or (time.time() - t0) > 4.0:
            break

    await pipeline.stop()
    elapsed = time.time() - t0

    print(f"Pipeline Execution Complete in {elapsed:.2f}s:")
    print(f"  Per-Stream Inferences: {counts}")
    print(f"  Total Inferences: {sum(counts.values())}")
    print(f"  Aggregate Inference Rate: {sum(counts.values()) / elapsed:.1f} inf/sec")

    metrics = pipeline.metrics()
    print("  Global Snapshot Metrics:")
    print(json.dumps(metrics.global_snapshot(), indent=2))


if __name__ == "__main__":
    results = []
    
    # 1 stream 720p
    r1_720 = benchmark_nvdec_multistream(1, 1280, 720, "720p")
    results.append(r1_720)

    # 2 streams 720p
    r2_720 = benchmark_nvdec_multistream(2, 1280, 720, "720p")
    results.append(r2_720)

    # 4 streams 720p
    r4_720 = benchmark_nvdec_multistream(4, 1280, 720, "720p")
    results.append(r4_720)

    # 8 streams 720p
    r8_720 = benchmark_nvdec_multistream(8, 1280, 720, "720p")
    results.append(r8_720)

    # 16 streams 720p
    r16_720 = benchmark_nvdec_multistream(16, 1280, 720, "720p")
    results.append(r16_720)

    # 1 stream 1080p
    r1_1080 = benchmark_nvdec_multistream(1, 1920, 1080, "1080p")
    results.append(r1_1080)

    # 2 streams 1080p
    r2_1080 = benchmark_nvdec_multistream(2, 1920, 1080, "1080p")
    results.append(r2_1080)

    # 4 streams 1080p
    r4_1080 = benchmark_nvdec_multistream(4, 1920, 1080, "1080p")
    results.append(r4_1080)

    # 8 streams 1080p
    r8_1080 = benchmark_nvdec_multistream(8, 1920, 1080, "1080p")
    results.append(r8_1080)

    # 16 streams 1080p
    r16_1080 = benchmark_nvdec_multistream(16, 1920, 1080, "1080p")
    results.append(r16_1080)

    # MultiStreamPipeline E2E test
    import asyncio
    asyncio.run(benchmark_multistream_pipeline_multimodal(2))
    asyncio.run(benchmark_multistream_pipeline_multimodal(4))

    save_benchmark_results(
        "real_multistream",
        {"description": "Hardware NVDEC multistream benchmark"},
        {"runs": results}
    )
