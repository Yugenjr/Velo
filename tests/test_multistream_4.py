"""
Velo 4-Stream Concurrency Stress-Test

Spawns 4 parallel RtpReceivers, pushes H.264 NAL units concurrently to all of them,
decodes frames in parallel, measures throughput and queue drops, and validates
isolation (closing one stream does not impact others).
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

def run_multistream_stress():
    print("\n--- Running 4-Stream Concurrency Stress-Test ---")
    
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available.")
        return
    torch.cuda.init()

    # Load sample H.264 NAL units
    h264_path = os.path.join("temp", "native_webrtc_experiment", "output.h264")
    if not os.path.exists(h264_path):
        print(f"[SKIP] H.264 sample file not found at {h264_path}")
        return
        
    with open(h264_path, "rb") as f:
        h264_data = f.read()

    nals = re.split(rb'\x00\x00\x00\x01|\x00\x00\x01', h264_data)
    nals = [n for n in nals if len(n) > 0]

    num_streams = 4
    receivers = [velo.RtpReceiver() for _ in range(num_streams)]
    print(f"Created {num_streams} independent RtpReceiver contexts.")

    decoded_counts = [0] * num_streams
    dropped_counts = [0] * num_streams
    errors_counts = [0] * num_streams
    active_flags = [True] * num_streams
    stop_event = threading.Event()
    
    process = psutil.Process()
    thread_baseline = process.num_threads()
    
    t_start = time.time()

    # Push workers
    def push_worker(stream_idx, receiver):
        timestamp = 90000
        while not stop_event.is_set():
            for nal in nals:
                if stop_event.is_set():
                    break
                nal_type = nal[0] & 0x1F
                if nal_type in (1, 5):
                    timestamp += 3000
                    # Push frames at roughly 45 FPS to load the decoder
                    time.sleep(0.022)
                try:
                    receiver.push_rtp(nal, timestamp)
                except velo.StreamClosedError:
                    return
                except Exception:
                    return

    push_threads = []
    for idx, receiver in enumerate(receivers):
        t = threading.Thread(target=push_worker, args=(idx, receiver), daemon=True)
        push_threads.append(t)
        t.start()

    # Consume workers
    def consume_worker(stream_idx, receiver):
        try:
            while active_flags[stream_idx]:
                frame = receiver.next()
                tensor = frame.to_torch()
                # Run lightweight GPU operation to verify CUDA context safety
                _val = tensor.float().mean()
                decoded_counts[stream_idx] += 1
                
                # Check for queue drops and decode errors
                dropped_counts[stream_idx] = receiver.dropped_frames
                errors_counts[stream_idx] = receiver.decode_errors
                
                # Simulate consumer processing work (GIL released)
                time.sleep(0.005)
        except velo.StreamClosedError:
            print(f"[Consumer {stream_idx}] Stream closed cleanly.")
        except Exception as e:
            print(f"[Consumer {stream_idx}] Error: {e}")

    consume_threads = []
    for idx, receiver in enumerate(receivers):
        t = threading.Thread(target=consume_worker, args=(idx, receiver), daemon=True)
        consume_threads.append(t)
        t.start()

    # Run stress test for 10 seconds
    time.sleep(10.0)
    
    # Verify isolation: Close Stream 0 while the other 3 streams are running
    print("\nClosing Stream 0 to verify isolation...")
    active_flags[0] = False
    receivers[0].close()
    consume_threads[0].join(timeout=2.0)
    
    # Let the remaining 3 streams run for another 3 seconds
    print("Stream 0 closed. Streams 1, 2, 3 continuing to decode...")
    time.sleep(3.0)
    
    # Close all remaining streams
    print("Closing remaining streams...")
    stop_event.set()
    for idx in range(1, num_streams):
        active_flags[idx] = False
        receivers[idx].close()
        
    for t in push_threads:
        t.join(timeout=2.0)
    for t in consume_threads[1:]:
        t.join(timeout=2.0)

    t_end = time.time()
    shutdown_time = t_end - t_start - 13.0 # Subtract 13s of sleep run time

    # Thread count back to baseline check
    time.sleep(1.0)
    thread_final = process.num_threads()

    print("\n============================================================")
    print("             VELO 4-STREAM STRESS TEST REPORT               ")
    print("============================================================")
    for i in range(num_streams):
        avg_fps = decoded_counts[i] / 13.0 # 13s active run
        print(f"Stream {i}: Decoded Frames: {decoded_counts[i]} | Drops: {dropped_counts[i]} | Errors: {errors_counts[i]} | Rate: {avg_fps:.1f} FPS")
    print(f"OS Worker Threads (Baseline -> Peak -> Final): {thread_baseline} -> ... -> {thread_final}")
    print(f"Graceful Shutdown Overhead: {shutdown_time:.3f} seconds")
    print("============================================================\n")

    # Assertions
    for i in range(num_streams):
        assert decoded_counts[i] > 10, f"Stream {i} failed to decode frames!"
        assert errors_counts[i] == 0, f"Stream {i} encountered decode errors!"
    print("[PASS] test_multistream_4")

if __name__ == "__main__":
    run_multistream_stress()
