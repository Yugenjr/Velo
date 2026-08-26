import torch
import torch.nn.functional as F
from typing import Union, List, Tuple
from .core import Frame

class GPUPreprocessor:
    """
    Model-agnostic GPU-native image preprocessing pipeline.
    Transforms NVDEC CUDA frames into VLM-ready PyTorch tensors
    without crossing the CPU boundary.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = (384, 384),
        mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
        std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711),
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize the preprocessor with standard VLM constraints.
        Defaults match SigLIP / Idefics2 / SmolVLM requirements.
        """
        self.target_size = target_size
        # Prepare mean and std for broadcast over (N, C, H, W)
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.dtype = dtype

    def preprocess(self, frames: Union[Frame, List[Frame]]) -> torch.Tensor:
        """
        Preprocess one or more frames purely on GPU.
        
        Args:
            frames: Single Frame or list of Frames.
            
        Returns:
            torch.Tensor: Shape (N, C, H, W) on GPU, formatted for VLM input.
        """
        if not isinstance(frames, list):
            frames = [frames]
            
        if not frames:
            raise ValueError("No frames provided for preprocessing.")
            
        processed_tensors = []
        device = None
        
        for frame in frames:
            # 1. Zero-copy NVDEC tensor retrieval (H, W, 4) uint8 on CUDA
            tensor = frame.to_torch()
            
            if device is None:
                device = tensor.device
                # Lazy initialization of constants to match frame device
                self.mean = self.mean.to(device=device, dtype=self.dtype)
                self.std = self.std.to(device=device, dtype=self.dtype)
                
            # 2. RGB Extraction & HWC -> CHW layout conversion
            if tensor.dim() == 3 and tensor.shape[-1] >= 3:
                tensor = tensor[..., :3]  # Extract RGB
                tensor = tensor.permute(2, 0, 1)  # (C, H, W)
            else:
                raise ValueError(f"Unexpected frame shape: {tensor.shape}")
                
            # 3. Add batch dimension (1, C, H, W) and cast to model dtype
            tensor = tensor.unsqueeze(0).to(self.dtype)
            
            # 4. Scale to [0, 1] range
            tensor = tensor / 255.0
            
            # 5. Resize via CUDA-accelerated bilinear interpolation
            if self.target_size is not None:
                tensor = F.interpolate(
                    tensor,
                    size=self.target_size,
                    mode='bilinear',
                    align_corners=False
                )
                
            # 6. Apply mean/std normalization
            tensor = (tensor - self.mean) / self.std
            
            processed_tensors.append(tensor)
            
        # 7. Batch concatenation
        if len(processed_tensors) == 1:
            return processed_tensors[0]
        else:
            return torch.cat(processed_tensors, dim=0)
