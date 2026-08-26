"""
Velo VLM Real Integration Benchmark

Attempts to run the full WebRTC -> NVDEC -> AIScheduler -> SmolVLM pipeline.
Will gracefully exit with BLOCKED status if sufficient VRAM or LiveKit credentials
are not available.
"""
import os
import sys
import time
import threading
import torch

import velo

print("=" * 60)
print("          VELO VLM INTEGRATION BENCHMARK              ")
print("=" * 60)

def generate_token(api_key: str, secret: str) -> str:
    from livekit import api
    return (
        api.AccessToken(api_key, secret)
        .with_identity("velo-vlm-bench")
        .with_grants(api.VideoGrants(room_join=True, room="velo-room"))
        .to_jwt()
    )

def run_benchmark():
    # 1. Hardware Check
    if not torch.cuda.is_available():
        print("\nREAL VLM BENCHMARK: BLOCKED / UNVERIFIED")
        print("CUDA is required for real VLM inference in Velo.")
        sys.exit(0)
        
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gb < 5.0:
        print(f"\nREAL VLM BENCHMARK: BLOCKED / UNVERIFIED")
        print(f"Insufficient VRAM: Detected {vram_gb:.2f} GB. Real VLM requires at least ~5 GB even in bfloat16.")
        # Graceful exit without failing the pipeline
        sys.exit(0)

    # 2. LiveKit Credential Check
    url = os.environ.get("LIVEKIT_URL")
    api_key = os.environ.get("LIVEKIT_API_KEY")
    secret = os.environ.get("LIVEKIT_API_SECRET")

    if not (url and api_key and secret):
        print("\nREAL VLM BENCHMARK: BLOCKED / UNVERIFIED")
        print("Missing LiveKit credentials (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET).")
        sys.exit(0)

    token = os.environ.get("LIVEKIT_TOKEN")
    if not token:
        try:
            token = generate_token(api_key, secret)
        except Exception as e:
            print(f"Failed to generate token: {e}")
            sys.exit(1)

    print(f"[INFO] Initializing SmolVLMAdapter (Loading Model)...")
    try:
        model_load_t0 = time.time()
        vlm = velo.SmolVLMAdapter()
        print(f"[SUCCESS] Model loaded in {time.time() - model_load_t0:.2f}s.")
    except Exception as e:
        print(f"[ERROR] Failed to load VLM: {e}")
        sys.exit(1)

    print(f"[INFO] Connecting to LiveKit at {url}...")
    try:
        connect_t0 = time.time()
        stream = velo.connect_livekit(url, token)
        connection_latency = time.time() - connect_t0
        print(f"[SUCCESS] Connected in {connection_latency:.3f}s.")
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        sys.exit(1)

    TARGET_FPS = 1.0 # 1 inference per second
    scheduler = velo.AIScheduler(target_fps=TARGET_FPS, temporal_window=5.0)
    
    producer_active = True

    def producer():
        try:
            while producer_active:
                frame = stream.next()
                scheduler.submit(frame)
        except velo.StreamClosedError:
            pass
        except Exception as e:
            print(f"Producer error: {e}")
        finally:
            scheduler.close()

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    print(f"[INFO] Starting E2E Inference Benchmark...")
    
    start_time = time.time()
    frames_inferred = 0
    duration = 15.0 # Run for 15 seconds
    
    total_inference_latency = 0.0
    total_preprocessing_latency = 0.0

    try:
        while time.time() - start_time < duration:
            try:
                # E2E Lifecycle
                frame = scheduler.acquire()
                
                # Single-frame inference
                response = vlm.generate(frame, prompt="Describe what is happening in a few words.")
                
                frames_inferred += 1
                total_inference_latency += response.latency_ms
                total_preprocessing_latency += response.preprocessing_latency_ms
                
                print(f"  [Inference {frames_inferred}] Latency: {response.latency_ms:.1f}ms - '{response.text}'")
                
                scheduler.release()
            except velo.SchedulerClosedError:
                break
    except KeyboardInterrupt:
        pass
    finally:
        producer_active = False
        stream.close()
        scheduler.close()
        producer_thread.join(timeout=2.0)

    stats = scheduler.stats()
    
    avg_inf = (total_inference_latency / frames_inferred) if frames_inferred > 0 else 0.0
    avg_pre = (total_preprocessing_latency / frames_inferred) if frames_inferred > 0 else 0.0
    
    print("\n" + "=" * 50)
    print("           VLM E2E BENCHMARK SUMMARY           ")
    print("=" * 50)
    print(f"Connection Latency: {connection_latency:.3f}s")
    print(f"Frames Received from LiveKit: {stats['frames_received']}")
    print(f"Frames Dropped (Stale/Backpressure): {stats['frames_dropped']}")
    print(f"Frames Processed via VLM: {frames_inferred}")
    print(f"Average Preprocessing Latency: {avg_pre:.2f} ms")
    print(f"Average Inference Latency: {avg_inf:.2f} ms")
    print(f"Average Total VLM Latency: {avg_pre + avg_inf:.2f} ms")
    print(f"Effective Inference FPS: {stats['effective_inference_fps']:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    run_benchmark()
