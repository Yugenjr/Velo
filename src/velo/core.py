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

    def __init__(self, native_stream):
        self._native = native_stream

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


def connect(sdp_offer: str):
    """
    Connect to a WebRTC peer and begin receiving GPU-decoded video.

    Args:
        sdp_offer: The SDP offer string from the browser.

    Returns:
        A tuple of (Stream, sdp_answer_string).
    """
    native_stream, sdp_answer = _velo_native.connect(sdp_offer)
    return Stream(native_stream), sdp_answer
