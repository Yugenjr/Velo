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
    tensors = []
    
    try:
        # Consume frames for up to 10 seconds or until 20 frames are processed
        while frame_count < 20 and (time.time() - start_time < 10.0):
            try:
                frame = stream.next()
                tensor = frame.to_torch()
                tensors.append(tensor)
                frame_count += 1
                if frame_count == 1:
                    print("[SUCCESS] First LiveKit frame decoded successfully!")
                    print(f"  Shape: {frame.shape}")
                    print(f"  Residency: {tensor.device}")
                    assert tensor.device.type == "cuda"
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
    print("\n" + "=" * 40)
    print("           VERIFICATION SUMMARY           ")
    print("=" * 40)
    print(f"Result: PASS" if frame_count > 0 else "Result: FAIL (No frames received)")
    print(f"Total Frames Received: {frame_count}")
    print(f"Sustained Decode Rate: {frame_count / duration:.2f} FPS" if duration > 0 else "N/A")
    print(f"Dropped Frames: {stream.dropped_frames}")
    print(f"Decode Errors: {stream.decode_errors}")
    print("=" * 40)

if __name__ == "__main__":
    run_real_verification()
