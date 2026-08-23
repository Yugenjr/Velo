import torch
from typing import Optional
from . import _velo_native
from .exceptions import StreamClosedError

class Frame:
    """
    A decoded video frame residing in GPU memory.
    """
    def __init__(self, dlpack_capsule, decoder_ref):
        self._capsule = dlpack_capsule
        self._decoder_ref = decoder_ref

    def to_torch(self) -> torch.Tensor:
        """
        Consumes the frame and returns a PyTorch CUDA tensor.
        The resulting tensor is on cuda:0 and shares memory with the NVDEC surface.
        """
        return torch.from_dlpack(self._capsule)

class Stream:
    """
    A WebRTC video stream.
    """
    def __init__(self, native_stream):
        self._native_stream = native_stream
        self._closed = False

    def next(self) -> Frame:
        """
        Blocks until the next frame is decoded and available.
        Returns a velo.Frame.
        Raises velo.StreamClosedError if the peer disconnected or the stream was closed.
        """
        if self._closed:
            raise StreamClosedError("Stream is already closed.")
        
        try:
            # Native blocking call, releases GIL internally
            capsule, decoder = self._native_stream.next_frame()
            return Frame(capsule, decoder)
        except Exception as e:
            if "StreamClosed" in str(e):
                self._closed = True
                raise StreamClosedError(str(e))
            raise

    def close(self):
        """
        Terminates the WebRTC connection and cleans up the native decoder.
        """
        if not self._closed:
            self._native_stream.close()
            self._closed = True


def connect(sdp_offer: str):
    """
    Connects to a WebRTC peer given an SDP offer.
    Returns a tuple of (Stream, sdp_answer_string).
    """
    native_stream, sdp_answer = _velo_native.connect(sdp_offer)
    return Stream(native_stream), sdp_answer
