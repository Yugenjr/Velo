import torch
from dataclasses import dataclass
from .audio import AudioChunk

@dataclass
class VoiceActivityResult:
    """
    Result of a Voice Activity Detection (VAD) evaluation.
    
    Attributes:
        is_speech (bool): True if speech was detected, False if silence.
        energy (float): The computed energy/RMS value of the chunk.
    """
    is_speech: bool
    energy: float

class VAD:
    """
    A lightweight, deterministic heuristic Voice Activity Detector (VAD)
    based on Root Mean Square (RMS) energy.
    """
    def __init__(self, energy_threshold: float = 0.01):
        """
        Args:
            energy_threshold: The RMS energy threshold above which audio is considered speech.
                              Depends heavily on input normalization (e.g., float audio in [-1.0, 1.0]).
        """
        self.energy_threshold = energy_threshold
        
    def analyze(self, chunk: AudioChunk) -> VoiceActivityResult:
        """
        Evaluate an AudioChunk for voice activity.
        """
        samples = chunk.samples
        if samples.numel() == 0:
            return VoiceActivityResult(is_speech=False, energy=0.0)
            
        # Convert to float32 for RMS calculation if not already
        if samples.dtype != torch.float32:
            samples = samples.to(torch.float32)
            
        # Calculate RMS energy: sqrt(mean(x^2))
        rms_energy = torch.sqrt(torch.mean(samples ** 2)).item()
        
        is_speech = rms_energy > self.energy_threshold
        
        return VoiceActivityResult(
            is_speech=is_speech,
            energy=rms_energy
        )
