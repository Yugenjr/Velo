import torch
import torch.nn.functional as F
from dataclasses import dataclass
from .core import Frame

@dataclass
class ChangeResult:
    changed: bool
    score: float
    threshold: float

class SceneChangeDetector:
    """
    A lightweight, GPU-native heuristic for detecting visual changes in video frames.
    
    This is NOT a semantic understanding model. It is a cheap signal to avoid
    redundant AI inference when the scene is visually static.
    """
    def __init__(self, threshold: float = 0.05, resolution: int = 64):
        """
        Args:
            threshold: The Mean Absolute Error (MAE) threshold above which a frame is considered "changed".
            resolution: The dimension to downsample the frame to (resolution x resolution).
        """
        self.threshold = threshold
        self.resolution = resolution
        self.reference_tensor = None
        
        # Luminance weights (0.299*R + 0.587*G + 0.114*B)
        self._luminance_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)

    def update(self, frame: Frame) -> ChangeResult:
        """
        Compare the current frame against the last reference frame.
        Updates the reference frame if a significant change is detected (or if it's the first frame).
        
        Args:
            frame: Velo Frame (CUDA resident).
            
        Returns:
            ChangeResult containing whether the frame changed and its score.
        """
        tensor = frame.to_torch()
        device = tensor.device
        
        # Ensure weights are on the correct device (lazy init)
        if self._luminance_weights.device != device:
            self._luminance_weights = self._luminance_weights.to(device)

        # 1. RGB Extraction & HWC -> CHW
        if tensor.dim() == 3 and tensor.shape[-1] >= 3:
            # NVDEC is typically uint8 RGBA. Convert to float32 NCHW RGB.
            tensor = tensor[..., :3].permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0
        else:
            raise ValueError(f"Unexpected frame shape: {tensor.shape}")

        # 2. Resize to low resolution for speed and noise reduction
        small = F.interpolate(tensor, size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)
        
        # 3. Convert to grayscale (luminance)
        lum = (small * self._luminance_weights).sum(dim=1, keepdim=True)

        # 4. Compare against reference
        if self.reference_tensor is None:
            self.reference_tensor = lum.detach()
            return ChangeResult(changed=True, score=1.0, threshold=self.threshold)
            
        # L1 distance (Mean Absolute Error)
        score = F.l1_loss(lum, self.reference_tensor).item()
        
        changed = score > self.threshold
        
        # Update accumulation baseline only when a change is detected.
        # This handles slow continuous panning gracefully.
        if changed:
            self.reference_tensor = lum.detach()
            
        return ChangeResult(changed=changed, score=score, threshold=self.threshold)
