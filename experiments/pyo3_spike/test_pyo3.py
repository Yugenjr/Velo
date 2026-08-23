import sys
import os
import shutil

# The rust DLL will be at target/release/velo_native.dll
dll_path = os.path.join("target", "release", "velo_native.dll")
pyd_path = "velo_native.pyd"

if os.path.exists(dll_path):
    shutil.copyfile(dll_path, pyd_path)
    
try:
    import velo_native
    print("SUCCESS: Imported velo_native")
    print(f"VERSION: {velo_native.version()}")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
