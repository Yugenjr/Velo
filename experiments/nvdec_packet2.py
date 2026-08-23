import sys
import torch

try:
    import PyNvVideoCodec as nvc
except ImportError as e:
    sys.exit(1)

p = nvc.PacketData()
print("bsl:", type(p.bsl), p.bsl)
print("bsl_data:", type(p.bsl_data))

try:
    p.bsl = 5
    print("Assigned bsl")
except Exception as e:
    print("Cannot assign bsl:", e)

try:
    p.bsl_data = b"hello"
    print("Assigned bsl_data with bytes")
except Exception as e:
    print("Cannot assign bytes to bsl_data:", e)

try:
    p.bsl_data = bytearray(b"hello")
    print("Assigned bsl_data with bytearray")
except Exception as e:
    print("Cannot assign bytearray to bsl_data:", e)
    
try:
    # See if there's an add_data or something, no, we printed dir
    pass
except Exception as e:
    pass

