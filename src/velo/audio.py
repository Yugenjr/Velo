import torch
from dataclasses import dataclass

@dataclass(frozen=True)
class AudioChunk:
    """
    An immutable representation of an audio chunk.
    
    Attributes:
        samples (torch.Tensor): PyTorch tensor holding the raw audio samples (typically shape [samples] for mono).
        sample_rate (int): The sample rate of the audio (e.g., 16000).
        channels (int): Number of audio channels (e.g., 1 for mono).
        timestamp (float): The start timestamp of the chunk in seconds.
        duration (float): The duration of the chunk in seconds.
    """
    samples: torch.Tensor
    sample_rate: int
    channels: int
    timestamp: float
    duration: float
    
    def __post_init__(self):
        # Validate data types to avoid unexpected issues downstream
        if not isinstance(self.samples, torch.Tensor):
            raise TypeError("samples must be a torch.Tensor")
        if not isinstance(self.sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if not isinstance(self.channels, int):
            raise TypeError("channels must be an integer")
        if not isinstance(self.timestamp, (int, float)):
            raise TypeError("timestamp must be a float")
        if not isinstance(self.duration, (int, float)):
            raise TypeError("duration must be a float")
