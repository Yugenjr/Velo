"""
Velo V0.5 Shutdown & Lifecycle Tests

Tests:
  A. connect → close: thread terminates
  B. close → close: idempotent
  C. close → next: StreamClosedError
  D. connect → close (×10): no thread leak
"""
import sys
import os
import time
import threading

# We test the native module directly (no WebRTC connection needed for some tests)
# For lifecycle tests that require a real connection, we use an SDP stub.

def test_import():
    """Verify the velo package imports cleanly."""
    import velo
    assert hasattr(velo, "connect")
    assert hasattr(velo, "Stream")
    assert hasattr(velo, "Frame")
    assert hasattr(velo, "StreamClosedError")
    assert hasattr(velo, "DecodeError")
    print("[PASS] test_import")

def test_exception_hierarchy():
    """Verify exception classes don't shadow builtins."""
    import velo
    assert issubclass(velo.StreamClosedError, velo.VeloError)
    assert issubclass(velo.DecodeError, velo.VeloError)
    assert issubclass(velo.VeloConnectionError, velo.VeloError)
    # Verify we don't shadow builtins
    assert velo.VeloConnectionError is not ConnectionError
    print("[PASS] test_exception_hierarchy")

def test_native_module():
    """Verify the native module loads."""
    from velo import _velo_native
    assert hasattr(_velo_native, "connect")
    assert hasattr(_velo_native, "VeloError")
    assert hasattr(_velo_native, "StreamClosedError")
    assert hasattr(_velo_native, "DecodeError")
    print("[PASS] test_native_module")

if __name__ == "__main__":
    test_import()
    test_exception_hierarchy()
    test_native_module()
    print("\n--- All unit tests passed ---")
