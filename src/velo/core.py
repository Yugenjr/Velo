import torch
from . import _velo_native
from .exceptions import StreamClosedError


class Frame:
    """
    A decoded video frame residing in GPU memory.

    Holds internal references to the DLPack capsule and the native decoder
    so the CUDA context outlives the tensor.
    """

    __slots__ = ("_capsule", "_decoder_ref")

    def __init__(self, dlpack_capsule, decoder_ref):
        self._capsule = dlpack_capsule
        self._decoder_ref = decoder_ref

    def to_torch(self) -> torch.Tensor:
        """
        Returns a PyTorch CUDA tensor backed by the GPU-resident frame.
        Zero-copy — shares memory with the NVDEC output surface.
        """
        return torch.from_dlpack(self._capsule)

    @property
    def shape(self):
        """Returns the shape of the decoded frame (height, width, channels)."""
        return self._capsule.shape

    @property
    def width(self) -> int:
        """Returns the width of the frame in pixels."""
        return self._capsule.shape[1]

    @property
    def height(self) -> int:
        """Returns the height of the frame in pixels."""
        return self._capsule.shape[0]

    @property
    def timestamp(self) -> float:
        """Returns the timestamp associated with this frame, if available."""
        try:
            return float(self._capsule.timestamp)
        except Exception:
            return 0.0


class Stream:
    """
    A live WebRTC video stream producing GPU-decoded frames.

    Lifecycle::

        stream, sdp_answer = velo.connect(sdp_offer)
        frame = stream.next()
        tensor = frame.to_torch()
        stream.close()
    """

    def __init__(self, native_stream, signaling_task=None):
        self._native = native_stream
        self._signaling_task = signaling_task

    def next(self) -> Frame:
        """
        Block until the next decoded frame is available.

        Releases the Python GIL internally so other threads can proceed.

        Raises:
            StreamClosedError: if the peer disconnected or close() was called.
        """
        try:
            capsule, decoder = self._native.next_frame()
            return Frame(capsule, decoder)
        except Exception as e:
            if "StreamClosed" in type(e).__name__ or "StreamClosed" in str(e):
                raise StreamClosedError(str(e)) from None
            raise

    def close(self):
        """
        Gracefully shut down the stream.

        - Signals the native worker thread to stop.
        - Joins the worker thread (releases GIL while waiting).
        - Drains remaining frames from the queue.
        - Releases WebRTC, NVDEC, and CUDA resources.

        Idempotent — safe to call multiple times.
        """
        try:
            self._native.close()
        except Exception:
            pass
        if hasattr(self, "_signaling_task") and self._signaling_task:
            try:
                self._signaling_task.cancel()
            except Exception:
                pass

    @property
    def dropped_frames(self) -> int:
        """Number of frames dropped by the bounded queue (drop-oldest policy)."""
        return self._native.dropped_frames

    @property
    def decode_errors(self) -> int:
        """Number of NVDEC decode errors encountered."""
        return self._native.decode_errors

    def __del__(self):
        try:
            self._native.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def connect(sdp_offer: str, max_width: int = None, max_height: int = None):
    """
    Connect to a WebRTC peer and begin receiving GPU-decoded video.

    Args:
        sdp_offer: The SDP offer string from the browser.
        max_width: Maximum expected stream width (default 1920).
        max_height: Maximum expected stream height (default 1080).

    Returns:
        A tuple of (Stream, sdp_answer_string).
    """
    native_stream, sdp_answer = _velo_native.connect(sdp_offer, max_width, max_height)
    return Stream(native_stream), sdp_answer


class RtpReceiver:
    """
    Ingests raw H.264 RTP packet payloads and decodes them directly to GPU memory.
    """

    def __init__(self, codec: str = "h264", max_width: int = None, max_height: int = None):
        self._native = _velo_native.RtpReceiver(codec, max_width, max_height)

    def push_rtp(self, payload: bytes, timestamp: int):
        """
        Push a raw H.264 RTP packet payload and its timestamp.
        """
        self._native.push_rtp(payload, timestamp)

    def next(self) -> Frame:
        """
        Block until the next decoded frame is available.
        """
        try:
            capsule, decoder = self._native.next_frame()
            return Frame(capsule, decoder)
        except Exception as e:
            if "StreamClosed" in type(e).__name__ or "StreamClosed" in str(e):
                raise StreamClosedError(str(e)) from None
            raise

    def close(self):
        """
        Gracefully shut down the receiver and join the worker thread.
        """
        self._native.close()

    @property
    def dropped_frames(self) -> int:
        """Number of frames dropped by the bounded queue (drop-oldest policy)."""
        return self._native.dropped_frames

    @property
    def decode_errors(self) -> int:
        """Number of NVDEC decode errors encountered."""
        return self._native.decode_errors

    def __del__(self):
        try:
            self._native.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def connect_livekit(url_or_track, token: str = None, codec: str = "h264"):
    """
    Connect to an incoming LiveKit video track.
    
    If url_or_track is a string, connects directly to the LiveKit SFU using WebSocket
    signaling and negotiates WebRTC to receive the stream.
    Otherwise, treats it as a mock track object yielding RTP packets for backwards compatibility.
    """
    if not isinstance(url_or_track, str):
        import threading
        import asyncio
        receiver = RtpReceiver(codec=codec)

        def thread_worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def ingest_loop():
                try:
                    async for packet in url_or_track.read_rtp():
                        receiver.push_rtp(packet.payload, packet.timestamp)
                except Exception:
                    pass
                finally:
                    try:
                        receiver.close()
                    except Exception:
                        pass

            loop.run_until_complete(ingest_loop())
            loop.close()

        t = threading.Thread(target=thread_worker, daemon=True)
        t.start()
        return receiver

    url = url_or_track
    import threading
    import asyncio
    import websockets
    import time
    from livekit.protocol import rtc as proto_rtc

    loop = asyncio.new_event_loop()
    sdp_offer_queue = asyncio.Queue()
    sdp_answer_queue = asyncio.Queue()
    shutdown_event = asyncio.Event()

    stream_container = []
    error_container = []

    def get_room_from_token(tok: str) -> str:
        try:
            import base64
            import json
            parts = tok.split('.')
            if len(parts) == 3:
                payload_b64 = parts[1]
                payload_b64 += '=' * (-len(payload_b64) % 4)
                payload = json.loads(base64.b64decode(payload_b64).decode('utf-8'))
                return payload.get('video', {}).get('room', '')
        except Exception:
            pass
        return ''

    async def signaling_task():
        ws_url = url
        if ws_url.startswith("http://"):
            ws_url = ws_url.replace("http://", "ws://", 1)
        elif ws_url.startswith("https://"):
            ws_url = ws_url.replace("https://", "wss://", 1)

        if not ws_url.endswith("/rtc"):
            ws_url = f"{ws_url.rstrip('/')}/rtc"

        room_name = get_room_from_token(token)
        ws_url = f"{ws_url}?sdk=python&version=1.1.14&protocol=15&access_token={token}"
        if room_name:
            ws_url = f"{ws_url}&room={room_name}"




        print(f"[DEBUG] ws_url: {ws_url}")
        try:
            async with websockets.connect(ws_url) as ws:

                async def ping_loop():
                    while not shutdown_event.is_set():
                        try:
                            await asyncio.sleep(15)
                            ping_req = proto_rtc.SignalRequest()
                            ping_req.ping = int(time.time() * 1000)
                            await ws.send(ping_req.SerializeToString())
                        except Exception:
                            break

                ping_task = loop.create_task(ping_loop())

                while not shutdown_event.is_set():
                    try:
                        msg = await ws.recv()
                        if not isinstance(msg, bytes):
                            continue

                        resp = proto_rtc.SignalResponse()
                        resp.ParseFromString(msg)

                        if resp.HasField("offer"):
                            sdp_offer = resp.offer.sdp
                            if not stream_container:
                                await sdp_offer_queue.put(sdp_offer)
                                sdp_answer = await sdp_answer_queue.get()
                            else:
                                native_stream = stream_container[0]
                                sdp_answer = await loop.run_in_executor(
                                    None, native_stream.set_offer, sdp_offer
                                )

                            answer_req = proto_rtc.SignalRequest()
                            answer_req.answer.sdp = sdp_answer
                            answer_req.answer.type = "answer"
                            await ws.send(answer_req.SerializeToString())

                        elif resp.HasField("pong"):
                            # Server replied to our ping, no action needed
                            pass


                    except websockets.exceptions.ConnectionClosed:
                        break
                    except Exception as e:
                        error_container.append(f"Signaling error: {e}")
                        break

                ping_task.cancel()
        except Exception as e:
            error_container.append(f"Connection failed: {e}")
            await sdp_offer_queue.put(None)

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(signaling_task())
        loop.close()

    t = threading.Thread(target=run_loop, daemon=True)
    t.start()

    sdp_offer = None
    t0 = time.time()
    while sdp_offer is None and time.time() - t0 < 15.0:
        if error_container:
            raise RuntimeError(f"LiveKit signaling error: {error_container[0]}")

        if not sdp_offer_queue.empty():
            sdp_offer = sdp_offer_queue.get_nowait()
        else:
            time.sleep(0.05)

    if sdp_offer is None:
        raise TimeoutError("Timed out waiting for LiveKit signaling connection or SDP offer.")

    native_stream, sdp_answer = _velo_native.connect(sdp_offer)
    stream_container.append(native_stream)

    loop.call_soon_threadsafe(sdp_answer_queue.put_nowait, sdp_answer)

    class SignalingTaskWrapper:
        def __init__(self, loop, shutdown_event):
            self.loop = loop
            self.shutdown_event = shutdown_event
        def cancel(self):
            if self.loop.is_running() and not self.loop.is_closed():
                try:
                    self.loop.call_soon_threadsafe(self.shutdown_event.set)
                except RuntimeError:
                    pass

    return Stream(native_stream, signaling_task=SignalingTaskWrapper(loop, shutdown_event))
