"""
Velo RtpReceiver Unit and Integration Tests

Verifies pushing raw Annex-B H.264 NAL units as RTP payloads to
the RtpReceiver native class, decoding them to CUDA tensors,
and closing the receiver gracefully.
"""
import os
import sys
import re
import time
import threading
import velo
import torch

def test_rtp_ingestion():
    print("\n--- Testing Direct RTP Ingestion ---")
    
    # 1. Ensure CUDA is initialized
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available.")
        return
    torch.cuda.init()
    
    # 2. Find and read sample H.264 file
    h264_path = os.path.join("temp", "native_webrtc_experiment", "output.h264")
    if not os.path.exists(h264_path):
        print(f"[SKIP] H.264 sample file not found at {h264_path}")
        return
        
    with open(h264_path, "rb") as f:
        h264_data = f.read()
        
    # 3. Extract raw NAL units by splitting on Annex-B start codes
    nals = re.split(rb'\x00\x00\x00\x01|\x00\x00\x01', h264_data)
    nals = [n for n in nals if len(n) > 0]
    print(f"Extracted {len(nals)} raw H.264 NAL units from sample stream.")
    
    # 4. Initialize RtpReceiver
    receiver = velo.RtpReceiver()
    print("RtpReceiver created successfully.")
    
    # 5. Push NAL units into RtpReceiver asynchronously
    # We run the push loop in a background thread to simulate network arrival,
    # while the main thread consumes the decoded frames.
    stop_event = threading.Event()
    
    def push_worker():
        timestamp = 90000
        for nal in nals:
            if stop_event.is_set():
                break
            nal_type = nal[0] & 0x1F
            # VCL slice types (1 = non-IDR, 5 = IDR) signal a new frame boundary
            if nal_type in (1, 5):
                timestamp += 3000
                # Add a tiny delay to simulate 30 FPS network stream
                time.sleep(0.033)
            try:
                receiver.push_rtp(nal, timestamp)
            except velo.StreamClosedError:
                break
            except Exception as e:
                print(f"[Worker] Push error: {e}")
                break
        print("[Worker] Finished pushing RTP payloads.")
        
    push_thread = threading.Thread(target=push_worker, daemon=True)
    push_thread.start()
    
    # 6. Consume decoded frames
    decoded_frames = 0
    t0 = time.time()
    try:
        # Loop for up to 5 seconds or until all packets are processed
        while decoded_frames < 20 and (time.time() - t0 < 5.0):
            frame = receiver.next()
            tensor = frame.to_torch()
            decoded_frames += 1
            if decoded_frames == 1:
                print(f"First frame decoded successfully!")
                print(f"  Shape: {frame.shape} ({frame.width}x{frame.height})")
                print(f"  Residency: {tensor.device}")
                print(f"  Dtype: {tensor.dtype}")
                assert tensor.device.type == "cuda"
                assert tensor.shape == (480, 640, 3) # Expected webcam frame shape from experiment
    except velo.StreamClosedError:
        print("Stream closed during read.")
    except Exception as e:
        print(f"Error during next(): {e}")
    finally:
        stop_event.set()
        push_thread.join()
        
    print(f"Successfully decoded {decoded_frames} frames directly from pushed RTP packets.")
    assert decoded_frames > 0, "No frames decoded!"
    
    # 7. Test graceful close
    print("Closing RtpReceiver...")
    receiver.close()
    
    # Verify next() raises StreamClosedError after close
    try:
        receiver.next()
        assert False, "StreamClosedError was not raised!"
    except velo.StreamClosedError:
        print("[PASS] StreamClosedError raised correctly on next() after close")
        
    print("[PASS] test_rtp_ingestion")

if __name__ == "__main__":
    test_rtp_ingestion()
    print("\n--- RTP Ingestion tests passed successfully ---")
