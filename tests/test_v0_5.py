"""
Velo V0.5 Integration Test — WebRTC → NVDEC → PyTorch Realtime Pipeline

Serves a browser page, receives WebRTC H.264, decodes via NVDEC,
and produces PyTorch CUDA tensors.

Also tests shutdown lifecycle.
"""
import os
import sys
import time
import threading
from aiohttp import web
import velo
import torch

# ---- CUDA check ----
if not torch.cuda.is_available():
    print("[FATAL] PyTorch CUDA not available. Exiting.")
    sys.exit(1)
torch.cuda.init()
print("[Python] PyTorch CUDA context initialized.")

# ---- State ----
stream = None
decode_thread = None
frames_processed = 0
start_time = None


def frame_loop():
    """Blocking frame consumer — runs in a background thread."""
    global frames_processed, start_time
    start_time = time.time()
    try:
        while True:
            frame = stream.next()
            tensor = frame.to_torch()
            frames_processed += 1

            if frames_processed % 30 == 0:
                elapsed = time.time() - start_time
                fps = frames_processed / elapsed if elapsed > 0 else 0
                print(
                    f"[Python] frames={frames_processed}  "
                    f"shape={tuple(tensor.shape)}  "
                    f"device={tensor.device}  "
                    f"dtype={tensor.dtype}  "
                    f"fps={fps:.1f}  "
                    f"dropped={stream.dropped_frames}  "
                    f"errors={stream.decode_errors}"
                )
    except velo.StreamClosedError:
        print("[Python] Stream closed.")
    except Exception as e:
        print(f"[Python] Error: {e}")


async def index(request):
    path = os.path.join(os.path.dirname(__file__), "index.html")
    html = open(path, "r").read()
    return web.Response(content_type="text/html", text=html)


async def offer(request):
    global stream, decode_thread
    params = await request.json()
    sdp = params["sdp"]

    print("[Python] SDP offer received. Connecting Velo native core...")
    try:
        stream, answer_sdp = velo.connect(sdp)
        print("[Python] Connected. Starting frame loop...")

        decode_thread = threading.Thread(target=frame_loop, daemon=True)
        decode_thread.start()

        return web.json_response({"sdp": answer_sdp, "type": "answer"})
    except Exception as e:
        print(f"[Python] Connect error: {e}")
        return web.Response(status=500, text=str(e))


async def shutdown(request):
    """Test endpoint: gracefully close the stream."""
    global stream
    if stream is not None:
        print("[Python] Closing stream via /shutdown endpoint...")
        t0 = time.time()
        stream.close()
        dt = time.time() - t0
        print(f"[Python] Stream closed in {dt:.3f}s.")
        print(f"[Python] Final stats: frames={frames_processed} "
              f"dropped={stream.dropped_frames} errors={stream.decode_errors}")
        stream = None
        return web.Response(text="closed")
    return web.Response(text="no active stream")


if __name__ == "__main__":
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_post("/shutdown", shutdown)

    print("=" * 50)
    print("  Velo V0.5 Integration Test Server")
    print("  POST /offer     — start WebRTC stream")
    print("  POST /shutdown  — test graceful close")
    print("=" * 50)
    web.run_app(app, host="0.0.0.0", port=8080)
