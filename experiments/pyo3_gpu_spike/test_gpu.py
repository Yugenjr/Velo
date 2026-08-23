import sys
import os
import shutil
import torch

# The rust DLL will be at target/release/velo_gpu.dll
dll_path = os.path.join("target", "release", "velo_gpu.dll")
pyd_path = "velo_gpu.pyd"

if os.path.exists(dll_path):
    shutil.copyfile(dll_path, pyd_path)
    
try:
    import velo_gpu
except ImportError as e:
    print(f"FAIL: Could not import velo_gpu: {e}")
    sys.exit(1)

def main():
    print("Starting V0.4 DLPack Boundary Test...")
    
    # Initialize PyTorch CUDA context first
    if torch.cuda.is_available():
        torch.cuda.init()
        print("[Python] PyTorch CUDA context initialized.")
    
    video_path = os.path.abspath(os.path.join("..", "..", "temp", "native_webrtc_experiment", "output.h264"))
    
    # 1. Call Rust function which reads the file, parses it, and decodes via NVDEC
    # returning a DecodedFrame (which natively supports DLPack via PyCapsule)
    try:
        print("[Python] Calling Rust velo_gpu.decode_webrtc_h264()...")
        frame, decoder = velo_gpu.decode_webrtc_h264(video_path)
    except Exception as e:
        print(f"FAIL: Rust decoding failed: {e}")
        sys.exit(1)

    print("[Python] Successfully received Python object from Rust!")
    print(f"[Python] Object Type: {type(frame)}")

    # 2. Convert to PyTorch Tensor via DLPack
    try:
        print("[Python] Converting to PyTorch tensor via torch.from_dlpack()...")
        tensor = torch.from_dlpack(frame)
    except Exception as e:
        print(f"FAIL: DLPack conversion failed: {e}")
        sys.exit(1)

    # 3. Print validation properties
    print("\n--- Result ---")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor device: {tensor.device}")

    if tensor.device.type == "cuda":
        print("\nGPU-native path: VERIFIED")
        print("CPU round-trip: NO")
    else:
        print("\nGPU-native path: NOT VERIFIED")
        print("CPU round-trip: YES")

if __name__ == "__main__":
    main()
