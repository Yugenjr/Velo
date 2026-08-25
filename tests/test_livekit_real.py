"""
Velo E2E LiveKit Real Connection & Verification Test

Attempts E2E room connection via WebSocket signaling and direct WebRTC negotiation
to feed raw H.264 RTP packets natively to NVDEC and output PyTorch CUDA tensors.
"""
import os
import sys
import time
import asyncio
import torch
import velo

print("=" * 60)
print("          VELO REAL LIVEKIT E2E VERIFICATION          ")
print("=" * 60)

def generate_token(api_key: str, secret: str) -> str:
    # pyrefly: ignore [missing-import]
    from livekit import api
    token = (
        api.AccessToken(api_key, secret)
        .with_identity("velo-verifier-real")
        .with_grants(api.VideoGrants(room_join=True, room="velo-room"))
        .to_jwt()
    )
    return token

def run_real_verification():
    url = os.environ.get("LIVEKIT_URL")
    api_key = os.environ.get("LIVEKIT_API_KEY")
    secret = os.environ.get("LIVEKIT_API_SECRET")

    # Audit credentials availability
    if not (url and api_key and secret):
        print("\nREAL LIVEKIT RUNTIME: BLOCKED / UNVERIFIED")
        print("\nReason for blocker: Missing environment variables:")
        if not url:
            print("  - LIVEKIT_URL is not set.")
        if not api_key:
            print("  - LIVEKIT_API_KEY is not set.")
        if not secret:
            print("  - LIVEKIT_API_SECRET is not set.")
        print("=" * 60)
        sys.exit(0)

    token = os.environ.get("LIVEKIT_TOKEN")
    if not token:
        print("[INFO] Generating Access Token...")
        try:
            token = generate_token(api_key, secret)
            print("[SUCCESS] Access Token generated.")
        except Exception as e:
            print(f"[ERROR] Failed to generate token: {e}")
            sys.exit(1)
    else:
        print("[SUCCESS] Using provided LIVEKIT_TOKEN.")


    print(f"[INFO] Connecting to LiveKit room via WebSocket signaling: {url}")
    t0 = time.time()
    try:
        stream = velo.connect_livekit(url, token)
        print(f"[SUCCESS] WebRTC subscription negotiated in {time.time() - t0:.3f} seconds.")
    except Exception as e:
        print(f"[ERROR] Connection / negotiation failed: {e}")
        sys.exit(1)

    print("[INFO] Waiting and receiving frames natively on GPU...")
    frame_count = 0
    start_time = time.time()
    first_frame_time = None
    last_frame_time = None
    inter_frame_intervals = []

    # Run for 60 seconds or until 1800 frames
    target_duration = 60.0
    target_frames = 1800

    try:
        while frame_count < target_frames and (time.time() - start_time < target_duration):
            try:
                frame = stream.next()
                tensor = frame.to_torch()
                t_after_next = time.time()
                frame_count += 1

                if frame_count == 1:
                    first_frame_time = t_after_next
                    print("[SUCCESS] First LiveKit frame decoded successfully!")
                    print(f"  Shape: {frame.shape}")
                    print(f"  Residency: {tensor.device}")
                    assert tensor.device.type == "cuda"
                else:
                    if last_frame_time:
                        inter_frame_intervals.append(t_after_next - last_frame_time)

                last_frame_time = t_after_next
            except velo.StreamClosedError:
                print("[INFO] Stream closed by peer.")
                break
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        print("[INFO] Closing LiveKit stream...")
        close_t0 = time.time()
        stream.close()
        close_time = time.time() - close_t0
        print(f"[SUCCESS] Stream closed in {close_time:.3f} seconds.")

    # Print results summary
    duration = time.time() - start_time
    setup_latency = (first_frame_time - start_time) if first_frame_time else duration

    steady_state_duration = (last_frame_time - first_frame_time) if (first_frame_time and last_frame_time and last_frame_time > first_frame_time) else 0.0
    steady_state_fps = (frame_count - 1) / steady_state_duration if steady_state_duration > 0.0 else 0.0
    wall_clock_fps = frame_count / duration if duration > 0.0 else 0.0

    avg_inter_frame = sum(inter_frame_intervals) / len(inter_frame_intervals) if inter_frame_intervals else 0.0

    print("\n" + "=" * 50)
    print("           VERIFICATION SUMMARY           ")
    print("=" * 50)
    print(f"Result: PASS" if frame_count > 0 else "Result: FAIL (No frames received)")
    print(f"Total Frames Received: {frame_count}")
    print(f"Connection Setup Latency: {setup_latency:.3f} seconds")
    print(f"Steady-State Decode Duration: {steady_state_duration:.3f} seconds")
    print(f"Steady-State Decode Rate: {steady_state_fps:.2f} FPS")
    print(f"Wall-Clock Decode Rate: {wall_clock_fps:.2f} FPS")
    print(f"Average Inter-Frame Interval: {avg_inter_frame * 1000:.2f} ms")
    print(f"Dropped Frames: {stream.dropped_frames}")
    print(f"Decode Errors: {stream.decode_errors}")
    print("=" * 50)

if __name__ == "__main__":
    run_real_verification()
