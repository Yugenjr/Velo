"""
Velo Multi-Stream Concurrency Tests

Verifies that multiple independent streams can be connected,
run concurrently, and closed gracefully without CUDA context conflicts.
"""
import sys
import time
import threading
import velo

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

def test_concurrency():
    print("\n--- Testing Multi-Stream Concurrency ---")
    
    # We will spawn 2 streams in parallel threads to simulate concurrent connection
    streams = []
    errors = []
    
    def connect_worker(id_):
        try:
            print(f"[Thread {id_}] Connecting stream...")
            stream, sdp_answer = velo.connect(dummy_sdp)
            print(f"[Thread {id_}] Stream connected successfully!")
            streams.append(stream)
        except Exception as e:
            print(f"[Thread {id_}] Connection error: {e}")
            errors.append(e)

    threads = []
    for i in range(2):
        t = threading.Thread(target=connect_worker, args=(i+1,), daemon=True)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    assert len(errors) == 0, f"Errors occurred during connection: {errors}"
    assert len(streams) == 2, f"Expected 2 active streams, got {len(streams)}"
    
    print("Both streams connected concurrently. Closing streams...")
    
    for i, stream in enumerate(streams):
        print(f"Closing stream {i+1}...")
        stream.close()
        
    print("[PASS] test_concurrency")

if __name__ == "__main__":
    test_concurrency()
    print("\n--- Multi-stream concurrency tests passed successfully ---")
