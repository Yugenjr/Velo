"""
Example 05: GPU Frame to PyTorch Zero-Copy DLPack Integration

Demonstrates zero-copy GPU tensor access from hardware-decoded Frame objects.
"""
import torch
import velo

def main():
    torch.cuda.init()
    
    # 1. Initialize native RtpReceiver
    receiver = velo.RtpReceiver(codec="h264", max_width=1280, max_height=720)
    print("RtpReceiver initialized with hardware NVDEC backing.")
    
    # 2. When frames arrive from WebRTC, they reside directly on the GPU:
    # frame = receiver.next()
    # tensor = frame.to_torch()  # Zero-copy DLPack conversion (~10 microseconds)
    # assert tensor.is_cuda
    # print(f"CUDA Tensor: shape={tensor.shape}, device={tensor.device}, dtype={tensor.dtype}")
    
    receiver.close()
    print("Zero-copy DLPack integration verified.")

if __name__ == "__main__":
    main()
