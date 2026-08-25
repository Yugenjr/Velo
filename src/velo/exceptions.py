from ._velo_native import (
    VeloError,
    VeloConnectionError,
    StreamClosedError,
    DecodeError,
)

class MediaError(VeloError):
    """Errors in the media stream (e.g., unsupported codec, packet parsing)."""
    pass


class HardwareError(VeloError):
    """Hardware decoding or CUDA errors."""
    pass
