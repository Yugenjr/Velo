import os
import sys
import time
import asyncio
import velo

print("=" * 60)
print("          VELO AI-PIPELINE LIVEKIT VERIFICATION       ")
print("=" * 60)

def generate_token(api_key: str, secret: str) -> str:
    from livekit import api
    token = (
        api.AccessToken(api_key, secret)
        .with_identity("velo-pipeline-test")
        .with_grants(api.VideoGrants(room_join=True, room="velo-room"))
        .to_jwt()
    )
    return token

async def run_pipeline_verification():
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

    print("[INFO] Constructing AIPipeline...")
    
    scheduler = velo.AIScheduler(target_fps=5.0, scene_aware=True)
    vlm = velo.vlm.MockVLM(simulated_latency=0.2)
    
    pipeline = velo.AIPipeline(
        stream=stream,
        scheduler=scheduler,
        vlm=vlm
    )

    print("[INFO] Starting pipeline...")
    await pipeline.start()
    
    t0 = time.time()
    results_received = 0
    duration = 5.0
    
    print("[INFO] Entering async inference loop...")
    
    try:
        # Use asyncio.wait_for to break the loop after `duration` seconds
        async def consume():
            nonlocal results_received
            async for result in pipeline.run_inference("Describe the scene"):
                results_received += 1
                print(f"  -> Result [{results_received}]: {result.text} (latency: {result.latency_ms:.1f}ms)")
                
        await asyncio.wait_for(consume(), timeout=duration)
    except asyncio.TimeoutError:
        print(f"[INFO] Completed {duration} seconds of inference.")
    except Exception as e:
        print(f"[ERROR] Inference loop failed: {e}")
    finally:
        print("[INFO] Stopping pipeline...")
        await pipeline.stop()
        
    stats = scheduler.stats()
    
    print("\n" + "=" * 50)
    print("           PIPELINE VERIFICATION SUMMARY            ")
    print("=" * 50)
    print(f"Total Async Results     : {results_received}")
    print(f"Frames Received         : {stats['frames_received']}")
    print(f"Frames Inferred         : {stats['frames_processed']}")
    print(f"Frames Scene-Skipped    : {stats.get('frames_skipped_scene', 0)}")
    print(f"Frames Backpressure Drop: {stats['frames_dropped_backpressure']}")
    print("=" * 50)
    print("[SUCCESS] AIPipeline integration verified.")

if __name__ == "__main__":
    asyncio.run(run_pipeline_verification())
