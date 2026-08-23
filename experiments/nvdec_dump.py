import sys
import torch

try:
    import PyNvVideoCodec as nvc
except ImportError as e:
    print(f"FAIL: PyNvVideoCodec is not installed: {e}")
    sys.exit(1)

print("CUDA and PyNvVideoCodec available.")

print("--- PyNvVideoCodec Module Dump ---")
for name in dir(nvc):
    if not name.startswith('_'):
        obj = getattr(nvc, name)
        print(f"{name}: {type(obj)}")

print("\n--- Inspecting specific classes ---")
for cls_name in ["Decoder", "PyNvDecoder", "DecoderSession", "Packet", "DecodeFrame", "NvidiaDecoder", "Decode", "SimpleDecoder", "CreateSimpleDecoder"]:
    if hasattr(nvc, cls_name):
        print(f"Found {cls_name}:")
        for method in dir(getattr(nvc, cls_name)):
            if not method.startswith('_'):
                print(f"  - {method}")
