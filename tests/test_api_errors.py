import sys
import unittest
import time
import velo
from velo.exceptions import VeloConnectionError, HardwareError, StreamClosedError
from velo import _velo_native

class TestAPIErrors(unittest.TestCase):

    def test_invalid_sdp(self):
        """Verify that connecting with an invalid SDP offer raises VeloConnectionError."""
        invalid_sdp = "v=0\r\no=invalid sdp\r\n"
        with self.assertRaises(VeloConnectionError):
            velo.connect(invalid_sdp)

    def test_invalid_codec(self):
        """Verify that instantiating RtpReceiver with invalid codecs raises ValueError."""
        # Planned but unsupported
        with self.assertRaises(ValueError) as ctx:
            velo.RtpReceiver(codec="vp8")
        self.assertIn("PLANNED but not yet supported", str(ctx.exception))

        # Completely invalid
        with self.assertRaises(ValueError) as ctx:
            velo.RtpReceiver(codec="invalid")
        self.assertIn("Unsupported or invalid codec", str(ctx.exception))

    def test_livekit_invalid_url(self):
        """Verify that connect_livekit with invalid URL or token times out or raises RuntimeError."""
        with self.assertRaises(Exception): # Usually TimeoutError or RuntimeError
            velo.connect_livekit("ws://127.0.0.1:9999", token="invalid_token")

    def test_next_on_closed_stream(self):
        """Verify reading from a closed RtpReceiver raises StreamClosedError."""
        receiver = velo.RtpReceiver()
        receiver.close()
        with self.assertRaises(StreamClosedError):
            receiver.next()

    def test_push_rtp_on_closed_receiver(self):
        """Verify pushing to a closed RtpReceiver raises StreamClosedError."""
        receiver = velo.RtpReceiver()
        receiver.close()
        with self.assertRaises(StreamClosedError):
            receiver.push_rtp(b"dummy_payload", 12345)

if __name__ == "__main__":
    unittest.main()
