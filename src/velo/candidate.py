import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from .core import Frame

@dataclass
class CandidateScore:
    """Standardized evaluation of whether a frame is a good inference candidate."""
    score: float
    accepted: bool
    reason: str


class CandidateSelector:
    """
    A lightweight, GPU-native heuristic for determining whether a visually changed
    frame is sufficiently informative to justify an expensive AI inference call.
    
    This operates strictly on visual signals (change magnitude, spatial variance, and temporal lockouts).
    It does not perform semantic understanding.
    """
    def __init__(self, threshold: float = 0.1, cooldown_ms: float = 1000.0, enabled: bool = True):
        self.threshold = threshold
        self.cooldown_ms = cooldown_ms
        self.enabled = enabled
        
        self.last_accepted_time = 0.0
        self.reference_tensor = None
        self.resolution = 64
        
        # Luminance weights (0.299*R + 0.587*G + 0.114*B)
        self._luminance_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)

    def analyze(self, frame: Frame) -> CandidateScore:
        if not self.enabled:
            return CandidateScore(score=1.0, accepted=True, reason="accepted")
            
        now = time.time() * 1000.0
        
        # 1. Ensure GPU residency and correct type
        tensor = frame.to_torch()
        device = tensor.device
        
        if self._luminance_weights.device != device:
            self._luminance_weights = self._luminance_weights.to(device)

        if tensor.dim() == 3 and tensor.shape[-1] >= 3:
            # Assuming NHWC input like NVDEC typically provides, extract RGB and make NCHW
            tensor = tensor[..., :3].permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0
        else:
            raise ValueError(f"Unexpected frame shape for candidate selection: {tensor.shape}")

        # 2. Extract cheap spatial/luminance features
        small = F.interpolate(tensor, size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)
        lum = (small * self._luminance_weights).sum(dim=1, keepdim=True)
        
        if self.reference_tensor is None:
            self.reference_tensor = lum.detach()
            self.last_accepted_time = now
            return CandidateScore(score=1.0, accepted=True, reason="novelty")

        # 3. Check temporal cooldown
        if (now - self.last_accepted_time) < self.cooldown_ms:
            return CandidateScore(score=0.0, accepted=False, reason="cooldown")

        # 4. Compute heuristic score
        # L1 difference against the LAST ACCEPTED frame
        change_magnitude = F.l1_loss(lum, self.reference_tensor).item()
        
        # Spatial variance ensures we don't trigger heavily on blank walls changing brightness slightly
        spatial_variance = torch.var(lum).item()
        
        # Multiply by 10.0 to scale it into roughly 0.0 - 1.0 range based on typical visual flows
        score = change_magnitude * (spatial_variance + 0.1) * 10.0
        
        if score > self.threshold:
            self.reference_tensor = lum.detach()
            self.last_accepted_time = now
            return CandidateScore(score=score, accepted=True, reason="scene_change")
            
        return CandidateScore(score=score, accepted=False, reason="low_information")
