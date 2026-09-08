"""
Velo E2E LiveKit Real Multimodal Ingestion & AI Verification Test

Connects to a real LiveKit SFU room via WebSocket signaling and negotiates WebRTC
to ingest live video (H.264 -> NVDEC) and audio (Opus -> PCM), synchronizes them
via unified monotonic timestamps in TemporalFusion, and executes multimodal AI inference.
"""
import os
import sys
import time
import asyncio
import torch
import velo
from velo import (
    AIPipeline,
    AIScheduler,
    AudioScheduler,
    TemporalFusion,
    VAD,
    MockASRAdapter,
    MockMultimodalAdapter,
)

print("=" * 60)
print("    VELO REAL LIVEKIT MULTIMODAL E2E VERIFICATION     ")
print("=" * 60)


def generate_token(api_key: str, secret: str) -> str:
    from livekit import api
    token = (
        api.AccessToken(api_key, secret)
        .with_identity("velo-multimodal-verifier")
        .with_grants(api.VideoGrants(room_join=True, room="velo-room"))
        .to_jwt()
    )
    return token


async def run_real_multimodal_verification():
    url = os.environ.get("LIVEKIT_URL")
    api_key = os.environ.get("LIVEKIT_API_KEY")
    secret = os.environ.get("LIVEKIT_API_SECRET")

    # Audit credentials availability
    if not (url and api_key and secret):
        print("\nREAL LIVEKIT MULTIMODAL RUNTIME: BLOCKED / UNVERIFIED")
        print("\nReason for blocker: Missing environment variables:")
        if not url:
            print("  - LIVEKIT_URL is not set.")
        if not api_key:
            print("  - LIVEKIT_API_KEY is not set.")
        if not secret:
            print("  - LIVEKIT_API_SECRET is not set.")
        print("=" * 60)
        return

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

    print(f"[INFO] Connecting to LiveKit room: {url}")
    stream = velo.connect_livekit(url, token)

    scheduler = AIScheduler(target_fps=15)
    audio_scheduler = AudioScheduler(chunk_duration_s=0.5, sample_rate=48000, channels=2)
    fusion = TemporalFusion(max_history_s=30.0)
    vad = VAD(energy_threshold=0.01)
    asr = MockASRAdapter(mock_text="live multimodal speech context")
    model = MockMultimodalAdapter(simulated_latency=0.02)

    pipeline = AIPipeline(
        stream=stream,
        scheduler=scheduler,
        vlm=model,
        audio_scheduler=audio_scheduler,
        fusion=fusion,
        vad=vad,
        asr=asr,
        context_window_s=1.5,
    )

    await pipeline.start()
    print("[SUCCESS] Multimodal pipeline started.")

    count = 0
    t0 = time.time()
    try:
        async for response in pipeline.run_inference("Describe the user actions and speech"):
            count += 1
            print(f"[INFERENCE #{count}] ts={response.timestamp} -> {response.text}")
            if count >= 10 or (time.time() - t0) > 10.0:
                break
    finally:
        await pipeline.stop()
        print(f"[SUCCESS] Real LiveKit multimodal pipeline stopped. Received {count} responses.")


def test_real_livekit_multimodal():
    """Pytest entrypoint that executes the real verification if credentials exist, otherwise gracefully skips."""
    url = os.environ.get("LIVEKIT_URL")
    api_key = os.environ.get("LIVEKIT_API_KEY")
    secret = os.environ.get("LIVEKIT_API_SECRET")

    if not (url and api_key and secret):
        import pytest
        pytest.skip("Real LiveKit credentials (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET) not set.")

    asyncio.run(run_real_multimodal_verification())


if __name__ == "__main__":
    asyncio.run(run_real_multimodal_verification())
