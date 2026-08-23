import sys
import torch

def test_gpu_decode():
    print("Initializing Velo V0.1-A Experiment")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available in PyTorch.")
        print(f"Current PyTorch version: {torch.__version__}")
        sys.exit(1)
        
    try:
        import PyNvVideoCodec as nvc
    except ImportError:
        print("FAIL: PyNvVideoCodec is not installed.")
        sys.exit(1)

    print("CUDA and PyNvVideoCodec available.")
    
    if len(sys.argv) < 2:
        print("Usage: python nvdec_pytorch.py <path-to-h264-or-mp4>")
        print("Please provide a valid video file path to run the experiment.")
        sys.exit(1)
        
    video_path = sys.argv[1]
    # 1. Initialize Decoder
    # Request GPU device memory output and specific color format suitable for PyTorch
    print("Opening decoder for", video_path)
    try:
        decoder = nvc.SimpleDecoder(
            video_path,
            gpu_id=0,
            use_device_memory=True, # Ensure frames stay in NVMM/GPU memory
            output_color_type=nvc.OutputColorType.RGB
        )
    except Exception as e:
        print(f"FAIL: Could not initialize decoder: {e}")
        sys.exit(1)
        
    # 2. Decode one frame
    print("Decoding one frame...")
    frames = decoder.get_batch_frames_by_index([0])
    
    if not frames:
        print("FAIL: No frames decoded.")
        sys.exit(1)
        
    frame = frames[0]
    
    # 3. Obtain DLPack tensor and convert to PyTorch
    print("Converting to PyTorch tensor via DLPack...")
    # torch.from_dlpack provides zero-copy access to the underlying DLManagedTensor
    # preventing a CPU round-trip.
    try:
        tensor = torch.from_dlpack(frame)
    except Exception as e:
        print(f"FAIL: DLPack conversion failed: {e}")
        sys.exit(1)
        
    # 4. Print results
    print("--- Result ---")
    print(f"Codec: H.264 (assumed from input)")
    print(f"Resolution: derived from tensor shape")
    print(f"Pixel format: RGB")
    print(f"Decoder: PyNvVideoCodec (NVDEC)")
    print(f"Device: GPU 0")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor device: {tensor.device}")
    
    # 5. Verify GPU residency
    if tensor.device.type == "cuda":
        print("\nGPU-native path: VERIFIED (via DLPack zero-copy)")
        print("CPU copy: NO")
    else:
        print("\nGPU-native path: NOT VERIFIED")
        print("CPU copy: YES (Tensor resides on CPU)")

if __name__ == "__main__":
    test_gpu_decode()
