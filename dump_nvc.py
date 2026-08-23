import os
import sys

# Try to set DLL directory if CUDA is in PATH
cuda_path = os.environ.get('CUDA_PATH', '')
if cuda_path and hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(os.path.join(cuda_path, "bin"))

try:
    import PyNvVideoCodec as nvc
except ImportError as e:
    print(f"FAILED TO IMPORT: {e}")
    sys.exit(1)

print("--- PyNvVideoCodec Module Dump ---")
for name in dir(nvc):
    if not name.startswith('_'):
        obj = getattr(nvc, name)
        print(f"{name}: {type(obj)}")

print("\n--- Inspecting specific classes ---")
for cls_name in ["Decoder", "PyNvDecoder", "DecoderSession", "Packet", "DecodeFrame", "NvidiaDecoder", "Decode"]:
    if hasattr(nvc, cls_name):
        print(f"Found {cls_name}:")
        for method in dir(getattr(nvc, cls_name)):
            if not method.startswith('_'):
                print(f"  - {method}")
