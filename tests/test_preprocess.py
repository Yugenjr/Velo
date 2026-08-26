import pytest
import torch
from velo.preprocess import GPUPreprocessor

class MockFrame:
    """Mock Velo Frame yielding a CUDA tensor mimicking NVDEC output."""
    def __init__(self, shape=(1080, 1920, 4)):
        self.shape = shape
        # NVDEC yields uint8 RGBA
        self._tensor = torch.randint(0, 255, shape, dtype=torch.uint8)
        
        if torch.cuda.is_available():
            self._tensor = self._tensor.to("cuda")

    def to_torch(self):
        return self._tensor

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU preprocessing tests")
def test_gpu_preprocess_single_frame():
    frame = MockFrame()
    preprocessor = GPUPreprocessor(target_size=(384, 384), dtype=torch.bfloat16)
    
    output = preprocessor.preprocess(frame)
    
    # Check shape: (Batch=1, C=3, H=384, W=384)
    assert output.shape == (1, 3, 384, 384)
    assert output.device.type == "cuda"
    assert output.dtype == torch.bfloat16

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU preprocessing tests")
def test_gpu_preprocess_temporal_frames():
    frames = [MockFrame() for _ in range(3)]
    preprocessor = GPUPreprocessor(target_size=(224, 224), dtype=torch.float32)
    
    output = preprocessor.preprocess(frames)
    
    # Check shape: (Batch=3, C=3, H=224, W=224)
    assert output.shape == (3, 3, 224, 224)
    assert output.device.type == "cuda"
    assert output.dtype == torch.float32
    
def test_cpu_fallback_if_no_cuda():
    """Test functionality on CPU if CUDA is unavailable."""
    frame = MockFrame()
    frame._tensor = frame._tensor.to("cpu")
    
    preprocessor = GPUPreprocessor(target_size=(384, 384), dtype=torch.float32)
    output = preprocessor.preprocess(frame)
    
    assert output.shape == (1, 3, 384, 384)
    assert output.device.type == "cpu"
    assert output.dtype == torch.float32
