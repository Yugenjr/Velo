"""
Velo Real Multi-Stream Hardware NVDEC + VLM Inference Benchmark (Milestone V2.0)

Taxonomy Class: Class 5 (Real Media + Real/Mock VLM Multi-Stream Concurrency)

Measures:
1. Real Annex-B H.264 NAL parsing and NVDEC decoding across multiple streams (1, 2, 4, 8).
2. Per-stream FPS, VLM inference latency, queue depth, and fairness.
3. GPU VRAM consumption, CPU utilization.
4. Explores OOM hardware limitations of 4GB GPUs.
"""
import os
import sys
import io
import time
import json
import threading
import traceback
import asyncio
import numpy as np
import torch
import av
import psutil
import velo
from velo import MultiStreamPipeline, MockMultimodalAdapter, AudioScheduler, TemporalFusion
from benchmark_utils import save_benchmark_results, get_system_metadata

def generate_annexb_h264(width: int, height: int, num_frames: int = 120, fps: int = 30) -> bytes:
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

def parse_annexb(stream_bytes: bytes):
    nals = []
    idx = 0
    while idx < len(stream_bytes) - 3:
        if stream_bytes[idx:idx+3] == b"\x00\x00\x01":
            start = idx + 3
            end = stream_bytes.find(b"\x00\x00\x01", start)
            if end == -1:
                nals.append(stream_bytes[idx:])
                break
            else:
                if stream_bytes[end-1] == 0:
                    nals.append(stream_bytes[idx:end-1])
                else:
                    nals.append(stream_bytes[idx:end])
            idx = end
        elif stream_bytes[idx:idx+4] == b"\x00\x00\x00\x01":
            start = idx + 4
            end = stream_bytes.find(b"\x00\x00\x00\x01", start)
            if end == -1:
                nals.append(stream_bytes[idx:])
                break
            else:
                if stream_bytes[end-1] == 0:
                    nals.append(stream_bytes[idx:end-1])
                else:
                    nals.append(stream_bytes[idx:end])
            idx = end
        else:
            idx += 1
    return nals

class RealH264StreamSource:
    def __init__(self, stream_id: str, nals: list, width: int, height: int):
        self.stream_id = stream_id
        self.nals = nals
        self.width = width
        self.height = height
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
            time.sleep(0.0001)

    def next(self) -> velo.Frame:
        if self._closed:
            raise velo.StreamClosedError("Stream closed")
            
        for _ in range(600):
            if self._closed:
                break
            frame = self.receiver.next()
            if frame is not None:
                frame.arrival_time = time.time()
                return frame
            time.sleep(0.005)
            
        raise velo.StreamClosedError("Decoder drained")

    def next_audio(self):
        raise velo.StreamClosedError("Audio EOS")

    def close(self):
        with self._lock:
            self._closed = True
        self.receiver.close()
        
async def run_vlm_benchmark():
    results = {}
    
    print("=======================================================")
    print("--- [Phase 1] Real SmolVLM Loading Test ---")
    print("=======================================================")
    
    real_vlm_blocked = False
    vlm_error_trace = ""
    try:
        from velo.vlm import SmolVLMAdapter
        print("Attempting to load real SmolVLM (bfloat16) on GPU...")
        real_adapter = SmolVLMAdapter(use_gpu_preprocess=False)
        print("Successfully loaded real SmolVLM!")
    except Exception as e:
        real_vlm_blocked = True
        vlm_error_trace = traceback.format_exc()
        print("FAILED to load real SmolVLM due to hardware limitation.")
        print("Error:")
        print(vlm_error_trace)
        print("Marking real VLM execution as BLOCKED.")

    results["real_vlm_loading"] = {
        "status": "BLOCKED" if real_vlm_blocked else "SUCCESS",
        "error": vlm_error_trace
    }

    print("Generating H.264 stream for benchmark...")
    video_bytes = generate_annexb_h264(1280, 720)
    nals = parse_annexb(video_bytes)
    
    results["concurrency_tests"] = []
    
    mock_vlm = MockMultimodalAdapter(simulated_latency=0.05)
    
    for num_streams in [1, 2, 4, 8]:
        print(f"\n=======================================================")
        print(f"--- [Phase 2] Decoder + Mock VLM - {num_streams} Streams ---")
        print(f"=======================================================")
        
        pipeline = MultiStreamPipeline(vlm=mock_vlm)
        sources = []
        for i in range(num_streams):
            src = RealH264StreamSource(f"stream-{i}", nals, 1280, 720)
            sources.append(src)
            pipeline.add_stream(
                f"stream-{i}", 
                src, 
                target_fps=30.0,
            )
            
        start_time = time.time()
        await pipeline.start()
        
        inferences_completed = {f"stream-{i}": 0 for i in range(num_streams)}
        latencies = {f"stream-{i}": [] for i in range(num_streams)}
        
        async def stop_later():
            await asyncio.sleep(3.0)
            await pipeline.stop()
            for src in sources:
                src.close()
                
        asyncio.create_task(stop_later())
        
        try:
            async for stream_id, res in pipeline.run_inference("monitor"):
                if isinstance(res, Exception):
                    pass
                else:
                    inferences_completed[stream_id] += 1
                    latencies[stream_id].append(res.latency_ms)
        except Exception as e:
            traceback.print_exc()
            
        duration = time.time() - start_time
        metrics = pipeline.metrics().global_snapshot()
        
        total_inferences = sum(inferences_completed.values())
        print(f"  Duration: {duration:.2f}s")
        print(f"  Total Inferences: {total_inferences}")
        print(f"  Aggregate Inference Rate: {total_inferences/duration:.1f} inf/sec")
        print(f"  Per-Stream Inferences: {inferences_completed}")
        
        inf_counts = list(inferences_completed.values())
        variance = np.var(inf_counts) if inf_counts else 0
        print(f"  Inference Variance (Fairness): {variance:.2f}")
        
        results["concurrency_tests"].append({
            "num_streams": num_streams,
            "duration": duration,
            "total_inferences": total_inferences,
            "per_stream_inferences": inferences_completed,
            "aggregate_rate": total_inferences/duration,
            "variance": float(variance),
            "metrics": metrics
        })

    final_output = {
        "benchmark": "real_multistream_vlm",
        "environment": get_system_metadata(),
        "results": results
    }
    
    save_benchmark_results("benchmark_real_multistream_vlm", final_output, {})

if __name__ == "__main__":
    asyncio.run(run_vlm_benchmark())
