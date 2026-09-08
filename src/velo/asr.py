from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional
from .audio import AudioChunk

@dataclass
class Transcript:
    """
    Result of Automatic Speech Recognition (ASR).
    
    Attributes:
        text (str): The recognized text.
        start_timestamp (float): The start timestamp of the utterance in seconds.
        end_timestamp (float): The end timestamp of the utterance in seconds.
        confidence (float): Confidence score between 0.0 and 1.0.
    """
    text: str
    start_timestamp: float
    end_timestamp: float
    confidence: float

class BaseASRAdapter(ABC):
    """
    Abstract interface for Automatic Speech Recognition (ASR).
    """
    @abstractmethod
    def transcribe(self, chunk: AudioChunk) -> Transcript:
        """
        Transcribe an AudioChunk into text.
        
        Args:
            chunk: The AudioChunk to transcribe.
            
        Returns:
            A Transcript object containing the recognized text and timing info.
        """
        pass

class MockASRAdapter(BaseASRAdapter):
    """
    A mock ASR adapter for testing integration without a real neural model.
    """
    def __init__(self, mock_text: str = "mock transcript", confidence: float = 0.99):
        self.mock_text = mock_text
        self.confidence = confidence
        
    def transcribe(self, chunk: AudioChunk) -> Transcript:
        # We assume processing time takes a tiny bit
        import time
        time.sleep(0.01)
        
        return Transcript(
            text=self.mock_text,
            start_timestamp=chunk.timestamp,
            end_timestamp=chunk.timestamp + chunk.duration,
            confidence=self.confidence
        )
