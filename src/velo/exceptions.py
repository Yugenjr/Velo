class VeloError(Exception):
    """Base class for all Velo exceptions."""
    pass


class VeloConnectionError(VeloError):
    """Failed to establish or maintain WebRTC connection."""
    pass


class MediaError(VeloError):
    """Errors in the media stream (e.g., unsupported codec, packet parsing)."""
    pass


class HardwareError(VeloError):
    """Hardware decoding or CUDA errors."""
    pass


class StreamClosedError(VeloError):
    """Raised when attempting to read from a closed stream or peer disconnected."""
    pass


class DecodeError(VeloError):
    """Raised when the NVDEC decoder encounters an error."""
    pass
