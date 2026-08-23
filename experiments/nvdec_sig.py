import sys
import torch
import inspect

try:
    import PyNvVideoCodec as nvc
except ImportError as e:
    sys.exit(1)

def print_sig(obj, name):
    try:
        sig = inspect.signature(obj)
        print(f"{name}{sig}")
    except Exception as e:
        print(f"Could not get signature for {name}: {e}")
        # Try getting docstring as fallback
        if obj.__doc__:
            print(f"{name} docstring:")
            print(obj.__doc__)

print_sig(nvc.CreateDecoder, "CreateDecoder")
print_sig(nvc.PyNvDecoder, "PyNvDecoder")
print_sig(nvc.PyNvDecoder.Decode, "PyNvDecoder.Decode")
print_sig(nvc.PyNvDemuxer, "PyNvDemuxer")
if hasattr(nvc, "Decode"):
    print_sig(nvc.Decode, "Decode")
