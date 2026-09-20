"""
Example 01: Video Stream Ingestion & Paced AI Scheduling

Demonstrates receiving live H.264 video frames via WebRTC/LiveKit,
pacing them with AIScheduler, and converting directly to PyTorch CUDA tensors.
"""
import velo
import torch

def main():
    torch.cuda.init()
    
    # 1. Connect to WebRTC or LiveKit video stream
    # stream = velo.connect_livekit("ws://localhost:7880", token="<JWT_TOKEN>")
    
    # 2. Configure an AI-aware temporal scheduler
    scheduler = velo.AIScheduler(
        target_fps=10.0,            # Pace input 30 FPS video down to 10 FPS for AI
        temporal_window=5.0,        # Retain 5 seconds of recent visual history
        scene_aware=True,           # Skip visually static frames on GPU
        scene_threshold=0.05,
        candidate_aware=True,       # Skip low-information frames
    )
    
    print("Velo AIScheduler configured. Ready to ingest frames zero-copy on CUDA.")

if __name__ == "__main__":
    main()
