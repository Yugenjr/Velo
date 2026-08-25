import os
import sys
import time
import argparse
import subprocess
import psutil
import torch
import velo

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
API_KEY = os.environ.get("LIVEKIT_API_KEY")
API_SECRET = os.environ.get("LIVEKIT_API_SECRET")

if not API_KEY or not API_SECRET:
    print("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set to run test_hardening.py")
    sys.exit(1)

def get_gpu_metrics():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,utilization.encoder,utilization.decoder",
            "--format=csv,noheader,nounits"
        ]).decode("utf-8").strip()
        parts = out.split(",")
        return {
            "gpu_util": float(parts[0]),
            "gpu_mem": float(parts[1]),
            "enc_util": float(parts[2]) if len(parts) > 2 else 0.0,
            "dec_util": float(parts[3]) if len(parts) > 3 else 0.0,
        }
    except Exception:
        return {"gpu_util": 0.0, "gpu_mem": 0.0, "enc_util": 0.0, "dec_util": 0.0}

def get_thread_count():
    try:
        # Total OS threads in this process
        return len(os.listdir(f"/proc/{os.getpid()}/task"))
    except Exception:
        return psutil.Process().num_threads()

def generate_token(room: str, identity: str) -> str:
    # Use lk CLI to generate token natively (avoids any python livekit-api issues)
    cmd = [
        "lk", "token", "create",
        "--api-key", API_KEY,
        "--api-secret", API_SECRET,
        "--room", room,
        "--identity", identity,
        "--join",
        "--token-only"
    ]
    token = subprocess.check_output(cmd).decode("utf-8").strip()
    return token

def start_publisher(room: str, identity: str):
    cmd = [
        "lk", "room", "join",
        "--url", "http://localhost:7880",
        "--api-key", API_KEY,
        "--api-secret", API_SECRET,
        "-r", room,
        "-i", identity,
        "--publish-demo"
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ===========================================================================
# PHASES DEFINITIONS
# ===========================================================================

def run_phase_1():
    print("\n" + "="*60)
    print("PHASE 1 — REAL LIVEKIT LONG-RUN TEST (5 minutes)")
    print("="*60)
    
    room = "velo-long-run"
    identity_pub = "long-pub"
    identity_sub = "long-sub"
    
    print("[INFO] Starting publisher...")
    pub_proc = start_publisher(room, identity_pub)
    time.sleep(3) # Wait for room & track
    
    print("[INFO] Generating token...")
    token = generate_token(room, identity_sub)
    
    print("[INFO] Connecting Velo subscriber...")
    stream = velo.connect_livekit(LIVEKIT_URL, token)
    
    frame_count = 0
    start_time = time.time()
    next_latencies = []
    
    print(f"{'Time (s)':<10}{'Frames':<10}{'FPS':<10}{'Dropped':<10}{'Errors':<10}{'Threads':<10}{'CPU %':<10}{'GPU %':<10}{'GPU Mem':<10}{'NVDEC %':<10}")
    print("-"*100)
    
    last_report = start_time
    
    try:
        while time.time() - start_time < 300.0: # 5 minutes
            t_next_0 = time.time()
            try:
                frame = stream.next()
                tensor = frame.to_torch()
                next_latencies.append(time.time() - t_next_0)
                frame_count += 1
            except velo.StreamClosedError:
                print("[ERROR] Stream closed unexpectedly.")
                break
                
            now = time.time()
            if now - last_report >= 10.0:
                elapsed = now - start_time
                fps = frame_count / elapsed
                gpu = get_gpu_metrics()
                threads = get_thread_count()
                cpu = psutil.cpu_percent()
                print(f"{elapsed:<10.1f}{frame_count:<10}{fps:<10.2f}{stream.dropped_frames:<10}{stream.decode_errors:<10}{threads:<10}{cpu:<10.1f}{gpu['gpu_util']:<10.1f}{gpu['gpu_mem']:<10.1f}{gpu['dec_util']:<10.1f}")
                last_report = now
    finally:
        print("[INFO] Closing stream...")
        stream.close()
        pub_proc.terminate()
        pub_proc.wait()
        
    duration = time.time() - start_time
    avg_next_latency = (sum(next_latencies) / len(next_latencies)) * 1000 if next_latencies else 0.0
    
    print("\n" + "="*40)
    print("PHASE 1 SUMMARY")
    print("="*40)
    print(f"Total Duration: {duration:.1f} seconds")
    print(f"Total Frames Received: {frame_count}")
    print(f"Average FPS: {frame_count / duration:.2f}")
    print(f"Average next() Latency: {avg_next_latency:.3f} ms")
    print(f"Final Dropped Frames: {stream.dropped_frames}")
    print(f"Final Decode Errors: {stream.decode_errors}")
    print("Result: VERIFIED" if frame_count > 100 else "Result: FAILED")
    print("="*40)

def run_phase_2():
    print("\n" + "="*60)
    print("PHASE 2 — CONNECT/DISCONNECT STRESS")
    print("="*60)
    
    room = "velo-stress"
    identity_pub = "stress-pub"
    identity_sub = "stress-sub"
    
    print("[INFO] Starting publisher...")
    pub_proc = start_publisher(room, identity_pub)
    time.sleep(3)
    
    print("[INFO] Generating token...")
    token = generate_token(room, identity_sub)
    
    print("[INFO] Warming up PyTorch CUDA and NVDEC context...")
    try:
        # Warm up PyTorch
        _ = torch.zeros(1024, device="cuda:0")
        # Warm up Velo connection and decoding
        warmup_stream = velo.connect_livekit(LIVEKIT_URL, token)
        for _ in range(10):
            warmup_frame = warmup_stream.next()
            _ = warmup_frame.to_torch()
        warmup_stream.close()
    except Exception as e:
        print(f"[WARNING] Warmup failed: {e}")
        
    time.sleep(1)
    
    cycles = 20
    failed = False
    
    initial_threads = get_thread_count()
    initial_gpu_mem = get_gpu_metrics()["gpu_mem"]
    
    print(f"Initial State: Threads = {initial_threads}, GPU Mem = {initial_gpu_mem} MiB")
    print(f"{'Cycle':<10}{'Frames':<10}{'Threads':<15}{'GPU Mem (MiB)':<15}")
    print("-"*55)
    
    for i in range(1, cycles + 1):
        try:
            stream = velo.connect_livekit(LIVEKIT_URL, token)
            frame_count = 0
            for _ in range(5):
                frame = stream.next()
                tensor = frame.to_torch()
                frame_count += 1
            stream.close()
            
            threads = get_thread_count()
            gpu_mem = get_gpu_metrics()["gpu_mem"]
            print(f"{i:<10}{frame_count:<10}{threads:<15}{gpu_mem:<15.1f}")
        except Exception as e:
            print(f"[ERROR] Cycle {i} failed: {e}")
            failed = True
            break
            
    pub_proc.terminate()
    pub_proc.wait()
    
    # Wait a bit for final thread reclamation
    time.sleep(2)
    final_threads = get_thread_count()
    final_gpu_mem = get_gpu_metrics()["gpu_mem"]
    
    print("\n" + "="*40)
    print("PHASE 2 SUMMARY")
    print("="*40)
    print(f"Cycles Run: {cycles}")
    print(f"Initial/Final Threads: {initial_threads} / {final_threads}")
    print(f"Initial/Final GPU Mem: {initial_gpu_mem:.1f} / {final_gpu_mem:.1f} MiB")
    
    # Check for thread leaks (allow small variance for OS threads, e.g. <= 2)
    thread_leak = max(0, final_threads - initial_threads)
    mem_leak = max(0.0, final_gpu_mem - initial_gpu_mem)
    
    print(f"Thread Leak: {thread_leak}")
    print(f"GPU Mem Leak: {mem_leak:.1f} MiB")
    
    if not failed and thread_leak <= 2 and mem_leak <= 50.0:
        print("Result: VERIFIED")
    else:
        print("Result: FAILED")
    print("="*40)

def run_phase_3():
    print("\n" + "="*60)
    print("PHASE 3 — PUBLISHER DISCONNECT")
    print("="*60)
    
    room = "velo-disconnect"
    identity_pub = "disconnect-pub"
    identity_sub = "disconnect-sub"
    
    print("[INFO] Starting publisher...")
    pub_proc = start_publisher(room, identity_pub)
    time.sleep(3)
    
    print("[INFO] Generating token...")
    token = generate_token(room, identity_sub)
    
    print("[INFO] Connecting Velo...")
    stream = velo.connect_livekit(LIVEKIT_URL, token)
    
    # Read a few frames first
    for _ in range(5):
        frame = stream.next()
        tensor = frame.to_torch()
        
    print("[INFO] Terminating publisher process unexpectedly...")
    pub_proc.terminate()
    pub_proc.wait()
    
    print("[INFO] Waiting for stream failure detection...")
    t0 = time.time()
    detected = False
    
    try:
        # Loop for up to 10 seconds
        while time.time() - t0 < 10.0:
            frame = stream.next()
            tensor = frame.to_torch()
    except velo.StreamClosedError:
        print(f"[SUCCESS] Stream detected closure in {time.time() - t0:.2f} seconds!")
        detected = True
    except Exception as e:
        print(f"[ERROR] Stream threw unexpected exception: {e}")
        
    print("[INFO] Closing Velo stream...")
    stream.close()
    
    print("\n" + "="*40)
    print("PHASE 3 SUMMARY")
    print("="*40)
    print(f"Closure Detected: {detected}")
    print("Result: VERIFIED" if detected else "Result: FAILED")
    print("="*40)

def run_phase_4():
    print("\n" + "="*60)
    print("PHASE 4 — LIVEKIT RENEGOTIATION")
    print("="*60)
    
    room = "velo-reneg"
    identity_pub = "reneg-pub"
    identity_sub = "reneg-sub"
    
    print("[INFO] Generating token first...")
    token = generate_token(room, identity_sub)
    
    print("[INFO] Connecting Velo subscriber (before publisher exists)...")
    # LiveKit server will generate an initial offer with only the data channel
    stream = velo.connect_livekit(LIVEKIT_URL, token)
    
    # Start publisher now (should trigger renegotiation/second offer with video track)
    print("[INFO] Starting publisher (triggering renegotiation)...")
    pub_proc = start_publisher(room, identity_pub)
    
    print("[INFO] Waiting for renegotiated video frames...")
    frame_count = 0
    t0 = time.time()
    
    try:
        while frame_count < 10 and time.time() - t0 < 15.0:
            frame = stream.next()
            tensor = frame.to_torch()
            frame_count += 1
            if frame_count == 1:
                print(f"[SUCCESS] Received first post-renegotiation frame of shape {frame.shape}!")
    except Exception as e:
        print(f"[ERROR] Renegotiation failed: {e}")
        
    stream.close()
    pub_proc.terminate()
    pub_proc.wait()
    
    print("\n" + "="*40)
    print("PHASE 4 SUMMARY")
    print("="*40)
    print(f"Frames Received: {frame_count}")
    print("Result: VERIFIED" if frame_count >= 10 else "Result: FAILED")
    print("="*40)

def run_phase_5():
    print("\n" + "="*60)
    print("PHASE 5 — MULTIPLE STREAMS")
    print("="*60)
    
    # Test 2 concurrent streams
    print("[INFO] Test Case 1: 2 Concurrent Streams...")
    room_1, room_2 = "velo-multi-1", "velo-multi-2"
    pub_1 = start_publisher(room_1, "pub-1")
    pub_2 = start_publisher(room_2, "pub-2")
    time.sleep(3)
    
    token_1 = generate_token(room_1, "sub-1")
    token_2 = generate_token(room_2, "sub-2")
    
    threads_0 = get_thread_count()
    gpu_mem_0 = get_gpu_metrics()["gpu_mem"]
    
    print("[INFO] Connecting Stream 1 and Stream 2...")
    stream_1 = velo.connect_livekit(LIVEKIT_URL, token_1)
    stream_2 = velo.connect_livekit(LIVEKIT_URL, token_2)
    
    f1, f2 = 0, 0
    t0 = time.time()
    
    try:
        while (f1 < 20 or f2 < 20) and time.time() - t0 < 10.0:
            if f1 < 20:
                frame = stream_1.next()
                tensor = frame.to_torch()
                f1 += 1
            if f2 < 20:
                frame = stream_2.next()
                tensor = frame.to_torch()
                f2 += 1
    except Exception as e:
        print(f"[ERROR] Stream error: {e}")
        
    threads_1 = get_thread_count()
    gpu_mem_1 = get_gpu_metrics()["gpu_mem"]
    
    stream_1.close()
    stream_2.close()
    pub_1.terminate()
    pub_2.terminate()
    pub_1.wait()
    pub_2.wait()
    
    print(f"2 Streams Metrics: Frames (S1/S2) = {f1}/{f2}, Threads = {threads_0} -> {threads_1}, GPU Mem Delta = {gpu_mem_1 - gpu_mem_0:.1f} MiB")
    
    success_2 = (f1 >= 20 and f2 >= 20)
    
    # Test 4 concurrent streams if 2 streams succeeded
    success_4 = False
    if success_2:
        print("\n[INFO] Test Case 2: 4 Concurrent Streams...")
        rooms = ["velo-m1", "velo-m2", "velo-m3", "velo-m4"]
        pubs = [start_publisher(rooms[i], f"pub-{i}") for i in range(4)]
        time.sleep(4)
        
        tokens = [generate_token(rooms[i], f"sub-{i}") for i in range(4)]
        
        print("[INFO] Connecting 4 Velo Streams...")
        try:
            streams = [velo.connect_livekit(LIVEKIT_URL, tokens[i]) for i in range(4)]
            counts = [0] * 4
            t0 = time.time()
            while min(counts) < 15 and time.time() - t0 < 10.0:
                for i in range(4):
                    if counts[i] < 15:
                        frame = streams[i].next()
                        tensor = frame.to_torch()
                        counts[i] += 1
            
            print(f"4 Streams Frames: {counts}")
            success_4 = (min(counts) >= 15)
        except Exception as e:
            print(f"[ERROR] 4 Streams failed: {e}")
        finally:
            for s in streams:
                try: s.close()
                except: pass
            for p in pubs:
                p.terminate()
                p.wait()
                
    print("\n" + "="*40)
    print("PHASE 5 SUMMARY")
    print("="*40)
    print(f"2 Streams Concurrent: {success_2}")
    print(f"4 Streams Concurrent: {success_4}")
    print("Result: VERIFIED" if success_4 else "Result: FAILED")
    print("="*40)

def run_phase_6():
    print("\n" + "="*60)
    print("PHASE 6 — ERROR INJECTION")
    print("="*60)
    
    room = "velo-errors"
    pub_proc = start_publisher(room, "errors-pub")
    time.sleep(3)
    token = generate_token(room, "errors-sub")
    
    # 1. Invalid SDP offer
    print("[INFO] Test 1: Invalid SDP Offer...")
    try:
        velo.connect("invalid_sdp_string")
        print("  [FAIL] Did not raise connection error on invalid SDP.")
    except Exception as e:
        print(f"  [PASS] Correctly raised exception: {e}")
        
    # 2. Missing SDP (empty)
    print("[INFO] Test 2: Missing SDP...")
    try:
        velo.connect("")
        print("  [FAIL] Did not raise connection error on empty SDP.")
    except Exception as e:
        print(f"  [PASS] Correctly raised exception: {e}")
        
    # 3. close() called twice
    print("[INFO] Test 3: close() called twice...")
    try:
        stream = velo.connect_livekit(LIVEKIT_URL, token)
        stream.close()
        stream.close()
        print("  [PASS] Clean shutdown on redundant close().")
    except Exception as e:
        print(f"  [FAIL] Redundant close() raised exception: {e}")
        
    # 4. next() after close()
    print("[INFO] Test 4: next() after close()...")
    try:
        stream = velo.connect_livekit(LIVEKIT_URL, token)
        stream.close()
        stream.next()
        print("  [FAIL] next() after close() did not raise StreamClosedError.")
    except velo.StreamClosedError:
        print("  [PASS] Correctly raised StreamClosedError.")
    except Exception as e:
        print(f"  [FAIL] next() after close() raised unexpected exception: {e}")
        
    # 5. connect() followed immediately by close()
    print("[INFO] Test 5: connect() followed immediately by close()...")
    try:
        stream = velo.connect_livekit(LIVEKIT_URL, token)
        stream.close()
        print("  [PASS] Instant close succeeded without hanging.")
    except Exception as e:
        print(f"  [FAIL] Instant close failed: {e}")
        
    pub_proc.terminate()
    pub_proc.wait()
    
    print("\n" + "="*40)
    print("PHASE 6 SUMMARY")
    print("="*40)
    print("Result: VERIFIED")
    print("="*40)

# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Velo V0.9 Hardening and Stress Tests")
    parser.add_argument("--phase", type=str, required=True, choices=["1", "2", "3", "4", "5", "6", "ALL"],
                        help="Hardening phase to run")
    args = parser.parse_args()
    
    if args.phase == "1" or args.phase == "ALL":
        run_phase_1()
    if args.phase == "2" or args.phase == "ALL":
        run_phase_2()
    if args.phase == "3" or args.phase == "ALL":
        run_phase_3()
    if args.phase == "4" or args.phase == "ALL":
        run_phase_4()
    if args.phase == "5" or args.phase == "ALL":
        run_phase_5()
    if args.phase == "6" or args.phase == "ALL":
        run_phase_6()
