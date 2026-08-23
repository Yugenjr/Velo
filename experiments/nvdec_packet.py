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
        if obj.__doc__:
            print(f"{name} docstring:")
            print(obj.__doc__)

print_sig(nvc.PacketData, "PacketData")
print_sig(nvc.PacketData.__init__, "PacketData.__init__")

# Let's see if we can create a PacketData from bytes
try:
    p = nvc.PacketData()
    print("Empty PacketData created.")
    print("PacketData dir:", dir(p))
except Exception as e:
    print("Could not create empty PacketData:", e)

try:
    p = nvc.PacketData(b'\x00')
    print("PacketData created from bytes.")
except Exception as e:
    print("Could not create PacketData from bytes:", e)

