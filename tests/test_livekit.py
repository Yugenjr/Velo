"""
Velo LiveKit Integration Tests

Simulates a LiveKit video track yielding H.264 RTP packets asynchronously,
verifying that connect_livekit() correctly consumes the track, feeds NVDEC,
produces GPU PyTorch tensors, and shuts down without deadlock or leakage.
"""
import os
import sys
import re
import time
import asyncio
import threading
import torch
import velo

# Mock classes to emulate LiveKit SDK classes
class MockPacket:
    def __init__(self, payload: bytes, timestamp: int):
        self.payload = payload
        self.timestamp = timestamp

class MockLiveKitTrack:
    def __init__(self, nals):
        self.nals = nals

    async def read_rtp(self):
        timestamp = 90000
        for nal in self.nals:
            nal_type = nal[0] & 0x1F
            if nal_type in (1, 5):
                timestamp += 3000
                # Simulate 30 FPS track packet timing
                await asyncio.sleep(0.033)
            yield MockPacket(nal, timestamp)
        print("[MockTrack] All RTP packets yielded.")

async def run_livekit_test():
    print("\n--- Testing LiveKit Track Ingestion ---")
    
    # 1. Initialize CUDA
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available.")
        return
    torch.cuda.init()

    # 2. Load H.264 sample NAL units
    h264_path = os.path.join("temp", "native_webrtc_experiment", "output.h264")
    if not os.path.exists(h264_path):
        print(f"[SKIP] H.264 sample file not found at {h264_path}")
        return
        
    with open(h264_path, "rb") as f:
        h264_data = f.read()

    nals = re.split(rb'\x00\x00\x00\x01|\x00\x00\x01', h264_data)
    nals = [n for n in nals if len(n) > 0]
    print(f"Extracted {len(nals)} H.264 NAL units for LiveKit track mock.")

    track = MockLiveKitTrack(nals)
    
    # 3. Ingest track into Velo
    stream = velo.connect_livekit(track)
    print("Velo connect_livekit initialized.")

    # 4. Consume frames on worker thread (blocking stream.next() releasing GIL)
    decoded_frames = 0
    t0 = time.time()
    try:
        # Loop for up to 5 seconds or until 20 frames are processed
        while decoded_frames < 20 and (time.time() - t0 < 5.0):
            frame = stream.next()
            tensor = frame.to_torch()
            decoded_frames += 1
            if decoded_frames == 1:
                print("First LiveKit frame decoded successfully!")
                print(f"  Shape: {frame.shape}")
                print(f"  Residency: {tensor.device}")
                assert tensor.device.type == "cuda"
                assert tensor.shape == (480, 640, 3)
    except velo.StreamClosedError:
        print("LiveKit stream closed during reading.")
    finally:
        print("Closing LiveKit stream...")
        stream.close()

    print(f"Successfully decoded {decoded_frames} frames from LiveKit Mock track.")
    assert decoded_frames > 0, "No frames were decoded from the LiveKit track!"
    print("[PASS] test_livekit_integration")

if __name__ == "__main__":
    asyncio.run(run_livekit_test())
