# Velo V1.0 Public API Reference

This document outlines the formal public API for the Velo package, as exposed through `velo.__init__.py`.

## Classes

### `velo.Frame`
A decoded video frame residing in GPU memory. It holds internal references to the DLPack capsule and the native decoder so the CUDA context outlives the tensor.

*   **Methods**
    *   `to_torch() -> torch.Tensor`: Returns a PyTorch CUDA tensor backed by the GPU-resident frame. Zero-copy — shares memory with the NVDEC output surface.
*   **Properties**
    *   `shape: tuple`: Returns the shape of the decoded frame (height, width, channels).
    *   `width: int`: Returns the width of the frame in pixels.
    *   `height: int`: Returns the height of the frame in pixels.
    *   `timestamp: float`: Returns the timestamp associated with this frame (0.0 if unavailable).

### `velo.Stream`
A live WebRTC video stream producing GPU-decoded frames. Can be used as a context manager (`with` block).

*   **Methods**
    *   `next() -> Frame`: Blocks until the next decoded frame is available. Releases the Python GIL internally. Raises `StreamClosedError` if the peer disconnected or `close()` was called.
    *   `close()`: Gracefully shuts down the stream, joins the worker thread, and releases resources. Idempotent.
*   **Properties**
    *   `dropped_frames: int`: Number of frames dropped by the bounded queue (drop-oldest policy).
    *   `decode_errors: int`: Number of NVDEC decode errors encountered.

### `velo.RtpReceiver`
Ingests raw H.264 RTP packet payloads and decodes them directly to GPU memory. Can be used as a context manager.

*   **Methods**
    *   `__init__(codec: str = "h264", max_width: int = None, max_height: int = None)`
    *   `push_rtp(payload: bytes, timestamp: int)`: Push a raw H.264 RTP packet payload and its timestamp.
    *   `next() -> Frame`: Blocks until the next decoded frame is available.
    *   `close()`: Gracefully shuts down the receiver and joins the worker thread.
*   **Properties**
    *   `dropped_frames: int`: Number of frames dropped by the bounded queue.
    *   `decode_errors: int`: Number of NVDEC decode errors encountered.

## Functions

### `velo.connect(sdp_offer: str, max_width: int = None, max_height: int = None) -> Tuple[Stream, str]`
Connects to a WebRTC peer and begins receiving GPU-decoded video.
*   **Parameters:**
    *   `sdp_offer` (str): The SDP offer string from the browser.
    *   `max_width` (int, optional): Maximum expected stream width (default 1920).
    *   `max_height` (int, optional): Maximum expected stream height (default 1080).
*   **Returns:** A tuple of `(Stream, sdp_answer)`.

### `velo.connect_livekit(url_or_track, token: str = None, codec: str = "h264") -> Union[Stream, RtpReceiver]`
Connects to an incoming LiveKit video track.
*   **Parameters:**
    *   `url_or_track` (str or object): The WebSocket URL to the LiveKit SFU, or a mock track object yielding RTP packets.
    *   `token` (str, optional): The LiveKit access token for WebSocket signaling.
    *   `codec` (str, optional): The codec to use (default "h264").
*   **Returns:** A `Stream` if a WebSocket URL is provided, or an `RtpReceiver` if a mock track is provided.

## Exceptions

All exceptions are exposed at the top level of the `velo` module.

*   **`VeloError`**: Base class for all Velo exceptions.
*   **`VeloConnectionError`**: Failed to establish or maintain WebRTC connection.
*   **`MediaError`**: Errors in the media stream (e.g., unsupported codec, packet parsing).
*   **`HardwareError`**: Hardware decoding or CUDA errors (e.g., missing CUDA toolkit or PyNvVideoCodec).
*   **`StreamClosedError`**: Raised when attempting to read from a closed stream or peer disconnected.
*   **`DecodeError`**: Raised when the NVDEC decoder encounters an error.
