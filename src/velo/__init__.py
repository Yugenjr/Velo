from .exceptions import (
    VeloError,
    VeloConnectionError,
    MediaError,
    HardwareError,
    StreamClosedError,
    DecodeError,
)

from .scheduler import AIScheduler, SchedulerClosedError
from .vlm import BaseVLMAdapter, MockVLM, SmolVLMAdapter, VLMResponse
from .preprocess import GPUPreprocessor
from .scene import SceneChangeDetector, ChangeResult
from .pipeline import AIPipeline, PipelineState

# Strict environment dependency audit at import time
try:
    import torch
except ImportError:
    raise ImportError(
        "Velo requires PyTorch with CUDA support.\n"
        "Please install it using:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
        "Verify with: python -c \"import torch; print(torch.cuda.is_available())\""
    ) from None

if not torch.cuda.is_available():
    raise HardwareError(
        "Velo requires a CUDA-capable NVIDIA GPU and a CUDA-enabled PyTorch installation.\n"
        "The currently installed PyTorch is CPU-only, which is incompatible with Velo's GPU-native pipeline.\n"
        "Please ensure NVIDIA drivers and CUDA Toolkit are installed, then reinstall PyTorch with CUDA support:\n"
        "  pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124"
    ) from None

try:
    import PyNvVideoCodec
except ImportError:
    raise HardwareError(
        "Velo requires PyNvVideoCodec.\n"
        "Please install the NVIDIA Video Codec SDK Python bindings:\n"
        "  pip install PyNvVideoCodec"
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
    "AIScheduler",
    "SchedulerClosedError",
    "BaseVLMAdapter",
    "MockVLM",
    "SmolVLMAdapter",
    "VLMResponse",
    "GPUPreprocessor",
    "SceneChangeDetector",
    "ChangeResult",
    "AIPipeline",
    "PipelineState",
]
