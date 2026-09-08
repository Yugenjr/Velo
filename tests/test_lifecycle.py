import sys
import time
import threading
import pytest
import velo
import torch

# H.264 SDP offer structure with valid ICE and fingerprint parameters
dummy_sdp = (
    "v=0\r\n"
    "o=- 1092644265492487431 2 IN IP4 127.0.0.1\r\n"
    "s=-\r\n"
    "t=0 0\r\n"
    "a=group:BUNDLE 0\r\n"
    "m=video 9 UDP/TLS/RTP/SAVPF 125\r\n"
    "c=IN IP4 127.0.0.1\r\n"
    "a=ice-ufrag:mockufrag\r\n"
    "a=ice-pwd:mockpwdmockpwdmockpwdmockpwdmock\r\n"
    "a=fingerprint:sha-256 00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF\r\n"
    "a=setup:actpass\r\n"
    "a=mid:0\r\n"
    "a=rtpmap:125 H264/90000\r\n"
    "a=fmtp:125 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f\r\n"
)

def get_current_thread_count():
    try:
        import psutil
        process = psutil.Process()
        return process.num_threads()
    except ImportError:
        return threading.active_count()

@pytest.fixture
def baseline_threads():
    stream, _ = velo.connect(dummy_sdp)
    stream.close()
    time.sleep(1.0)
    return get_current_thread_count()

def test_idempotent_close(baseline_threads):
    print("\n--- Testing Idempotent Close ---")
    print(f"Baseline thread count: {baseline_threads}")
    
    print("Connecting stream...")
    stream, sdp_answer = velo.connect(dummy_sdp)
    
    threads_after_connect = get_current_thread_count()
    print(f"Thread count after connect: {threads_after_connect}")
    
    print("Closing stream (1)...")
    stream.close()
    
    time.sleep(1.0) # Give worker thread time to exit and clean up
    threads_after_close = get_current_thread_count()
    print(f"Thread count after close: {threads_after_close}")
    
    assert threads_after_close <= baseline_threads, f"Thread leaked! Baseline: {baseline_threads}, got {threads_after_close}"
    
    print("Closing stream again (2)...")
    stream.close() # Should be safe and idempotent
    
    print("Closing stream again (3)...")
    stream.close()
    
    print("[PASS] test_idempotent_close")

def test_next_after_close():
    print("\n--- Testing next() After close() ---")
    stream, sdp_answer = velo.connect(dummy_sdp)
    stream.close()
    
    try:
        stream.next()
        assert False, "StreamClosedError was not raised!"
    except velo.StreamClosedError:
        print("[PASS] StreamClosedError raised correctly after close")
    except Exception as e:
        assert False, f"Unexpected exception: {e}"

def test_repeated_lifecycle(baseline_threads):
    print("\n--- Testing Repeated Lifecycle (Leak Test) ---")
    print(f"Baseline OS thread count: {baseline_threads}")
    
    for i in range(10):
        stream, _ = velo.connect(dummy_sdp)
        time.sleep(0.1)
        stream.close()
        time.sleep(0.1)
        current_threads = get_current_thread_count()
        print(f"Iteration {i+1}/10: Thread count = {current_threads}")
        
    # Final cooldown
    time.sleep(1.5)
    final_threads = get_current_thread_count()
    print(f"Final OS thread count: {final_threads} (baseline: {baseline_threads})")
    
    assert final_threads <= baseline_threads, f"Thread leak detected! Baseline: {baseline_threads}, Final: {final_threads}"
    print("[PASS] test_repeated_lifecycle (No thread leaks)")

if __name__ == "__main__":
    # Ensure psutil is installed
    try:
        import psutil
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil

    # 1. Warm-up iteration to load PyNvVideoCodec, CUDA contexts, and static pools
    print("Warming up Velo libraries...")
    stream, _ = velo.connect(dummy_sdp)
    stream.close()
    time.sleep(1.0)
    
    # 2. Capture baseline thread count AFTER warm-up
    baseline = get_current_thread_count()
    
    # 3. Run tests
    test_idempotent_close(baseline)
    test_next_after_close()
    test_repeated_lifecycle(baseline)
    
    print("\n--- All Lifecycle tests passed successfully ---")
