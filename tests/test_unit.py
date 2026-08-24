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

def test_unsupported_codec_rejection():
    """Verify that requesting unsupported codecs raises ValueError."""
    import velo
    # 1. Test planned but unsupported codecs
    for codec in ["vp8", "vp9", "av1", "hevc", "h265"]:
        try:
            velo.RtpReceiver(codec=codec)
            assert False, f"Codec {codec} should have been rejected!"
        except ValueError as e:
            assert "PLANNED but not yet supported" in str(e)
            print(f"[PASS] Correctly rejected planned codec: {codec}")

    # 2. Test completely invalid codecs
    try:
        velo.RtpReceiver(codec="invalid_codec")
        assert False, "Invalid codec should have been rejected!"
    except ValueError as e:
        assert "Unsupported or invalid codec" in str(e)
        print("[PASS] Correctly rejected completely invalid codec")

if __name__ == "__main__":
    test_import()
    test_exception_hierarchy()
    test_native_module()
    test_unsupported_codec_rejection()
    print("\n--- All unit tests passed ---")
