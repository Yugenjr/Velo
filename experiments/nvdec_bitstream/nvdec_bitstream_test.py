import sys
import ctypes
import torch

try:
    import PyNvVideoCodec as nvc
except ImportError as e:
    print("FAIL: PyNvVideoCodec is not installed.")
    sys.exit(1)

def test_direct_bitstream():
    print("Initializing Velo V0.2-D Experiment: Direct Bitstream -> NVDEC")
    
    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available in PyTorch.")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python nvdec_bitstream_test.py <path-to-h264>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    
    # 1. Initialize PyNvDecoder directly (NO Demuxer, NO SimpleDecoder)
    print("Creating native PyNvDecoder for direct packet input...")
    try:
        decoder = nvc.CreateDecoder(
            gpuid=0,
            codec=nvc.cudaVideoCodec.H264,
            usedevicememory=True,
            outputColorType=nvc.OutputColorType.RGB
        )
    except Exception as e:
        print(f"FAIL: Could not initialize direct decoder: {e}")
        sys.exit(1)
        
    # 2. Read raw bitstream
    print(f"Reading raw H.264 bitstream from {video_path}...")
    try:
        with open(video_path, 'rb') as f:
            data = f.read()
    except Exception as e:
        print(f"FAIL: Could not read bitstream: {e}")
        sys.exit(1)
        
    # 3. Create a raw C-buffer and assign to PacketData
    print("Constructing raw PacketData...")
    buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    packet = nvc.PacketData()
    packet.bsl = len(data)
    packet.bsl_data = ctypes.addressof(buffer)
    
    # 4. Decode the packet directly
    print("Submitting raw PacketData to NVDEC...")
    try:
        frames = decoder.Decode(packet)
    except Exception as e:
        print(f"FAIL: Decode failed: {e}")
        sys.exit(1)
        
    if not frames:
        print("FAIL: No frames decoded from the bitstream.")
        # Sometimes NVDEC needs an empty packet to flush
        print("Attempting to flush decoder...")
        empty_packet = nvc.PacketData()
        empty_packet.bsl = 0
        empty_packet.bsl_data = 0
        frames = decoder.Decode(empty_packet)
        if not frames:
            print("FAIL: Still no frames decoded.")
            sys.exit(1)
            
    print(f"SUCCESS: Decoded {len(frames)} frames directly from bitstream!")
    frame = frames[0]
    
    # 5. Convert to PyTorch tensor via DLPack
    print("Converting to PyTorch tensor via DLPack...")
    try:
        tensor = torch.from_dlpack(frame)
    except Exception as e:
        print(f"FAIL: DLPack conversion failed: {e}")
        sys.exit(1)
        
    # 6. Verify and Report
    print("\n--- Result ---")
    print("Codec: H264")
    print("Resolution: derived from tensor shape")
    print("Pixel format: RGB")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor device: {tensor.device}")
    
    if tensor.device.type == "cuda":
        print("\nGPU-native decode: VERIFIED")
        print("CPU copy: NO")
    else:
        print("\nGPU-native decode: NOT VERIFIED")
        print("CPU copy: YES (Tensor resides on CPU)")

if __name__ == "__main__":
    test_direct_bitstream()
