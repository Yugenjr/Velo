import argparse
import asyncio
import json
import logging
from aiohttp import web
# pyrefly: ignore [missing-import]
from aiortc import RTCPeerConnection, RTCSessionDescription

logging.basicConfig(level=logging.INFO)

async def handle_offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()

    @pc.on("track")
    def on_track(track):
        logging.info("Track received: %s", track.kind)
        
        if track.kind == "video":
            asyncio.ensure_future(inspect_video_track(track))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

async def inspect_video_track(track):
    print("--- Video Track Inspection Started ---")
    frame_count = 0
    try:
        while True:
            # recv() pulls the next frame from the aiortc jitter buffer / PyAV decoder
            frame = await track.recv()
            
            if frame_count == 0:
                print("\n--- FIRST FRAME INSPECTION ---")
                print(f"Frame type: {type(frame)}")
                print(f"Frame class: {frame.__class__.__name__}")
                print(f"Width: {frame.width}")
                print(f"Height: {frame.height}")
                print(f"Pixel format: {frame.format.name}")
                print(f"PTS: {frame.pts}")
                print(f"Time base: {frame.time_base}")
                print("------------------------------\n")
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Received {frame_count} frames (CPU Decoded)")
                
    except Exception as e:
        print(f"Track inspection ended: {e}")

import os

async def index(request):
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    return web.FileResponse(index_path)

app = web.Application()
app.router.add_get("/", index)
app.router.add_post("/offer", handle_offer)
app.router.add_static("/static", ".")  # Serve other static files if needed

if __name__ == "__main__":
    print("Starting V0.1-B WebRTC Receiver on http://localhost:8080")
    web.run_app(app, port=8080)
