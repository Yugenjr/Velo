import asyncio
import os
import sys
import time
import threading
from aiohttp import web
import velo
import torch

# Ensure CUDA is initialized for PyTorch and NVDEC
if not torch.cuda.is_available():
    print("[FATAL] PyTorch CUDA is not available. Velo requires an NVIDIA GPU.")
    sys.exit(1)
torch.cuda.init()
print("[Python] PyTorch CUDA context initialized.")

stream = None
decode_thread = None

async def index(request):
    path = os.path.join(os.path.dirname(__file__), "index.html")
    try:
        html = open(path, "r").read()
        return web.Response(content_type="text/html", text=html)
    except FileNotFoundError:
        return web.Response(status=404, text="index.html not found")

def frame_processing_loop():
    global stream
    print("[Python] Started blocking frame processing loop (GIL released).")
    frames_processed = 0
    start_time = time.time()
    
    try:
        while True:
            # Block until next frame is decoded natively
            frame = stream.next()
            
            # Map directly to PyTorch CUDA tensor (zero-copy)
            tensor = frame.to_torch()
            frames_processed += 1
            
            if frames_processed % 30 == 0:
                elapsed = time.time() - start_time
                fps = frames_processed / elapsed if elapsed > 0 else 0
                print(
                    f"[Velo Example] Frame {frames_processed} | "
                    f"Shape: {frame.shape} ({frame.width}x{frame.height}) | "
                    f"Residency: {tensor.device} | "
                    f"FPS: {fps:.1f} | "
                    f"Dropped: {stream.dropped_frames}"
                )
    except velo.StreamClosedError:
        print("[Python] Stream closed gracefully.")
    except Exception as e:
        print(f"[Python] Error in frame loop: {e}")
    finally:
        print("[Python] Consumer thread terminated.")

async def offer(request):
    global stream, decode_thread
    params = await request.json()
    sdp = params["sdp"]
    
    print("[Python] Received WebRTC offer. Connecting native Velo core...")
    try:
        # Establish connection. Returns Velo stream and local SDP answer.
        stream, answer_sdp = velo.connect(sdp)
        print("[Python] SDP negotiation successful. Launching frame consumer...")
        
        decode_thread = threading.Thread(target=frame_processing_loop, daemon=True)
        decode_thread.start()
        
        return web.json_response({
            "sdp": answer_sdp,
            "type": "answer"
        })
    except Exception as e:
        print(f"[Python] Connection failed: {e}")
        return web.Response(status=500, text=str(e))

async def shutdown(request):
    global stream
    if stream is not None:
        print("[Python] Gracefully closing Velo stream...")
        t0 = time.time()
        stream.close()
        print(f"[Python] Stream closed in {time.time() - t0:.3f} seconds.")
        stream = None
        return web.Response(text="Stream closed successfully.")
    return web.Response(text="No active stream to close.")

if __name__ == "__main__":
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_post("/shutdown", shutdown)
    
    print("=" * 60)
    print("  Velo 0.1.0 Basic Webcam Example Server")
    print("  1. Open http://localhost:8080 in your browser")
    print("  2. Click 'Start Streaming to Velo'")
    print("  3. Check console logs for GPU tensor shape and device")
    print("=" * 60)
    
    web.run_app(app, host="0.0.0.0", port=8080)
