"""
Velo E2E LiveKit Real Audio Ingestion & Verification Test

Connects to a real LiveKit SFU room via WebSocket signaling and negotiates WebRTC
to ingest live Opus audio RTP packets, decode them natively to 16-bit PCM in Rust,
and yield PyTorch-backed AudioChunk objects in Python.
"""
import os
import sys
import time
import torch
import velo
from velo.audio import AudioChunk
from velo.audio_scheduler import AudioScheduler

print("=" * 60)
print("       VELO REAL LIVEKIT AUDIO E2E VERIFICATION       ")
print("=" * 60)


def generate_token(api_key: str, secret: str) -> str:
    from livekit import api
    token = (
        api.AccessToken(api_key, secret)
        .with_identity("velo-audio-verifier")
        .with_grants(api.VideoGrants(room_join=True, room="velo-room"))
        .to_jwt()
    )
    return token


def run_real_audio_verification():
    url = os.environ.get("LIVEKIT_URL")
    api_key = os.environ.get("LIVEKIT_API_KEY")
    secret = os.environ.get("LIVEKIT_API_SECRET")

    # Audit credentials availability
    if not (url and api_key and secret):
        print("\nREAL LIVEKIT AUDIO RUNTIME: BLOCKED / UNVERIFIED")
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

    print("[INFO] Waiting and receiving audio chunks natively via stream.next_audio()...")
    audio_scheduler = AudioScheduler(chunk_duration_s=0.5, sample_rate=48000, channels=2)
    chunk_count = 0
    start_time = time.time()
    first_chunk_time = None
    last_chunk_time = None
    inter_chunk_intervals = []

    target_duration = 30.0
    target_chunks = 1500  # 30s of 20ms chunks

    try:
        while chunk_count < target_chunks and (time.time() - start_time < target_duration):
            try:
                chunk = stream.next_audio()
                t_after_next = time.time()
                chunk_count += 1
                audio_scheduler.submit(chunk)

                if chunk_count == 1:
                    first_chunk_time = t_after_next
                    print("[SUCCESS] First LiveKit audio chunk received & decoded!")
                    print(f"  Sample Rate: {chunk.sample_rate} Hz")
                    print(f"  Channels: {chunk.channels}")
                    print(f"  Timestamp: {chunk.timestamp:.4f} s")
                    print(f"  Duration: {chunk.duration:.4f} s")
                    print(f"  Samples dtype: {chunk.samples.dtype}, shape: {chunk.samples.shape}")
                    assert chunk.sample_rate == 48000
                    assert chunk.channels == 2
                else:
                    if last_chunk_time:
                        inter_chunk_intervals.append(t_after_next - last_chunk_time)

                last_chunk_time = t_after_next
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

    duration = time.time() - start_time
    setup_latency = (first_chunk_time - start_time) if first_chunk_time else duration
    avg_inter_chunk = sum(inter_chunk_intervals) / len(inter_chunk_intervals) if inter_chunk_intervals else 0.0

    print("\n" + "=" * 50)
    print("        AUDIO VERIFICATION SUMMARY        ")
    print("=" * 50)
    print(f"Result: {'PASS' if chunk_count > 0 else 'FAIL (No audio chunks received)'}")
    print(f"Total Audio Chunks Received: {chunk_count}")
    print(f"Connection Setup Latency: {setup_latency:.3f} seconds")
    print(f"Average Inter-Chunk Interval: {avg_inter_chunk * 1000:.2f} ms")
    print(f"Scheduler Stats: {audio_scheduler.stats()}")
    print("=" * 50)


if __name__ == "__main__":
    run_real_audio_verification()
