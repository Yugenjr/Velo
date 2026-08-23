import asyncio
import os
import sys
from aiohttp import web
import velo
import torch

# Initialize PyTorch CUDA context
if torch.cuda.is_available():
    torch.cuda.init()
    print("[Python] PyTorch CUDA context initialized.")
else:
    print("[Python] PyTorch CUDA not available. Exiting.")
    sys.exit(1)

# Global variables for testing
stream = None
decode_task = None

async def index(request):
    content = open(os.path.join(os.path.dirname(__file__), 'index.html'), 'r').read()
    return web.Response(content_type='text/html', text=content)

def frame_processing_loop():
    global stream
    print("[Python] Started blocking frame processing loop.")
    frames_processed = 0
    try:
        while True:
            # Blocks until the next frame is ready, releasing the GIL
            frame = stream.next()
            
            # Convert to PyTorch tensor natively via DLPack
            tensor = frame.to_torch()
            frames_processed += 1
            
            if frames_processed % 30 == 0:
                print(f"[Python] Processed {frames_processed} frames. Current tensor shape: {tensor.shape} on {tensor.device}")
                
    except velo.StreamClosedError:
        print("[Python] Stream closed by peer.")
    except Exception as e:
        print(f"[Python] Error in decode loop: {e}")
    finally:
        print("[Python] Frame loop terminated.")

async def offer(request):
    global stream, decode_task
    params = await request.json()
    sdp = params['sdp']
    
    print("[Python] Received SDP offer. Connecting native WebRTC core...")
    try:
        # Velo connect
        stream, answer_sdp = velo.connect(sdp)
        print("[Python] Velo connected. Starting decoder loop in background thread...")
        
        # We run the synchronous decode loop in a background executor 
        # so it doesn't block the asyncio event loop handling the HTTP server
        import threading
        decode_task = threading.Thread(target=frame_processing_loop, daemon=True)
        decode_task.start()
        
        return web.json_response({
            'sdp': answer_sdp,
            'type': 'answer'
        })
    except Exception as e:
        print(f"Error during connect: {e}")
        return web.Response(status=500, text=str(e))

if __name__ == '__main__':
    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_post('/offer', offer)
    
    print("==============================================")
    print(" Velo V0.5 Milestone Test Server ")
    print("==============================================")
    web.run_app(app, host='0.0.0.0', port=8080)
