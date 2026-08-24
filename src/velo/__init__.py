from .exceptions import (
    VeloError,
    VeloConnectionError,
    MediaError,
    HardwareError,
    StreamClosedError,
    DecodeError,
)

# Strict environment dependency audit at import time
try:
    import torch
except ImportError:
    raise ImportError("Velo requires PyTorch. Please install torch with CUDA support.") from None

if not torch.cuda.is_available():
    raise HardwareError(
        "Velo requires a CUDA-capable NVIDIA GPU. torch.cuda.is_available() is False. "
        "Please ensure NVIDIA drivers and CUDA Toolkit are installed."
    ) from None

try:
    import PyNvVideoCodec
except ImportError:
    raise HardwareError(
        "Velo requires PyNvVideoCodec. Please install the NVIDIA Video Codec SDK Python bindings."
    ) from None

from .core import connect, Stream, Frame, RtpReceiver, connect_livekit

__all__ = [
    "connect",
    "Stream",
    "Frame",
    "RtpReceiver",
    "connect_livekit",
    "VeloError",
    "VeloConnectionError",
    "MediaError",
    "HardwareError",
    "StreamClosedError",
    "DecodeError",
]
