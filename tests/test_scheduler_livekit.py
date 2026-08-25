"""
Velo AIScheduler Real LiveKit Integration Test

Connects to a LiveKit stream, feeds frames to AIScheduler,
and measures the incoming rate vs the scheduled rate.
"""
import os
import sys
import time
import threading
import velo

print("=" * 60)
print("          VELO AI-SCHEDULER LIVEKIT VERIFICATION      ")
print("=" * 60)

def generate_token(api_key: str, secret: str) -> str:
    from livekit import api
    token = (
        api.AccessToken(api_key, secret)
        .with_identity("velo-scheduler-test")
        .with_grants(api.VideoGrants(room_join=True, room="velo-room"))
        .to_jwt()
    )
    return token

def run_scheduler_verification():
    url = os.environ.get("LIVEKIT_URL")
    api_key = os.environ.get("LIVEKIT_API_KEY")
    secret = os.environ.get("LIVEKIT_API_SECRET")

    if not (url and api_key and secret):
        print("\nREAL LIVEKIT RUNTIME: BLOCKED / UNVERIFIED")
        print("Missing credentials.")
        sys.exit(0)

    token = os.environ.get("LIVEKIT_TOKEN")
    if not token:
        try:
            token = generate_token(api_key, secret)
        except Exception as e:
            print(f"Failed to generate token: {e}")
            sys.exit(1)

    print(f"[INFO] Connecting to LiveKit at {url}...")
    try:
        stream = velo.connect_livekit(url, token)
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        sys.exit(1)

    TARGET_FPS = 5.0
    scheduler = velo.AIScheduler(target_fps=TARGET_FPS, max_queue=2, strategy="latest")
    
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

    print(f"[INFO] Processing frames at target FPS: {TARGET_FPS}...")
    
    start_time = time.time()
    frames_processed = 0
    duration = 10.0 # Run for 10 seconds

    try:
        while time.time() - start_time < duration:
            try:
                frame = scheduler.next()
                tensor = frame.to_torch()
                frames_processed += 1
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
    
    print("\n" + "=" * 50)
    print("           SCHEDULER VERIFICATION SUMMARY           ")
    print("=" * 50)
    print(f"Target FPS: {TARGET_FPS:.2f}")
    print(f"Frames Received (Incoming): {stats['frames_received']}")
    print(f"Frames Processed (Yielded): {stats['frames_processed']}")
    print(f"Frames Dropped: {stats['frames_dropped']}")
    print(f"Average Queue Depth: {stats['average_queue_depth']:.2f}")
    print(f"Average Frame Age (Latency): {stats['average_frame_age'] * 1000:.2f} ms")
    print(f"Effective Processed FPS: {stats['effective_fps']:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    run_scheduler_verification()
