import time
from typing import Union, List
from .core import Frame


class VLMResponse:
    """Standardized response from a VLM adapter."""
    def __init__(self, text: str, latency_ms: float = 0.0, preprocessing_latency_ms: float = 0.0):
        self.text = text
        self.latency_ms = latency_ms
        self.preprocessing_latency_ms = preprocessing_latency_ms


class BaseVLMAdapter:
    """
    Abstract base class for VLM adapters.
    
    Decouples the specific multimodal AI model implementation from Velo's
    GPU-native scheduling pipeline.
    """
    def generate(self, frames: Union[Frame, List[Frame]], prompt: str) -> VLMResponse:
        """
        Run inference on a single frame or a temporal sequence of frames.
        
        Args:
            frames: A single velo.Frame or a list of velo.Frame objects.
            prompt: The text prompt describing what the model should look for.
            
        Returns:
            VLMResponse containing the generated text and latency metrics.
        """
        raise NotImplementedError


class MockVLM(BaseVLMAdapter):
    """
    A lightweight mock VLM for unit testing.
    Simulates inference latency without downloading models or requiring heavy VRAM.
    """
    def __init__(self, simulated_latency: float = 0.1):
        self.simulated_latency = simulated_latency
        
    def generate(self, frames: Union[Frame, List[Frame]], prompt: str) -> VLMResponse:
        t0 = time.time()
        
        if isinstance(frames, list):
            num_frames = len(frames)
        else:
            num_frames = 1
            
        time.sleep(self.simulated_latency)
        
        text = f"Mock response for {num_frames} frames with prompt: '{prompt}'"
        
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000
        
        return VLMResponse(text=text, latency_ms=latency_ms, preprocessing_latency_ms=0.0)


class SmolVLMAdapter(BaseVLMAdapter):
    """
    Adapter for HuggingFaceTB/SmolVLM-Instruct (2B parameters).
    
    Current Limitations:
    - Zero-copy GPU inference is currently broken during the preprocessing step 
      because the standard HuggingFace image processor strictly requires PIL 
      images or numpy arrays on the CPU.
    - This adapter explicitly converts GPU tensors to CPU PIL images to satisfy
      the transformer pipeline.
    """
    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM-Instruct", use_gpu_preprocess: bool = False):
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_gpu_preprocess = use_gpu_preprocess
        
        # Load in bfloat16 to minimize VRAM footprint (aiming to fit in 4GB).
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            _fast_init=False
        ).to(self.device)
        
        if self.use_gpu_preprocess:
            from .preprocess import GPUPreprocessor
            # Initialize with SmolVLM standard shapes (usually 384x384 for SigLIP base)
            self.gpu_preprocessor = GPUPreprocessor(
                target_size=(384, 384),
                dtype=torch.bfloat16
            )

    def _frame_to_pil(self, frame: Frame):
        from PIL import Image
        
        # CPU conversion required for standard HF processor
        tensor = frame.to_torch() # GPU tensor
        
        # Ensure channel layout is standard if needed. Velo native outputs NHWC (e.g. RGBA).
        if tensor.dim() == 3 and tensor.shape[2] == 4:
            # Convert RGBA to RGB
            tensor = tensor[:, :, :3]
            
        array = tensor.cpu().numpy()
        return Image.fromarray(array)

    def generate(self, frames: Union[Frame, List[Frame]], prompt: str) -> VLMResponse:
        import time
        import torch
        
        t0 = time.time()
        
        if not isinstance(frames, list):
            frames = [frames]
            
        if not frames:
            raise ValueError("At least one frame is required for VLM inference.")
            
        if self.use_gpu_preprocess:
            # PATH B: GPU Preprocessing (Zero-Copy)
            # 1. Preprocessing (GPU -> CUDA Tensor)
            # SmolVLM/Idefics2 natively expects pixel_values of shape (batch, num_images, num_patches, c, h, w)
            # or it handles simpler unpatched inputs depending on version. We supply the normalized tensor.
            # However, AutoProcessor text template is still required for the prompt.
            
            pixel_values = self.gpu_preprocessor.preprocess(frames)
            
            # Since Idefics2 requires specific nested patching, if we just pass a simple tensor, 
            # we must reshape it to match the expected format: (batch=1, num_images=len(frames), patches=1, C=3, H=384, W=384)
            pixel_values = pixel_values.unsqueeze(0).unsqueeze(2) # (1, num_images, 1, 3, 384, 384)
            
            # Still need text tokens from processor
            content = [{"type": "image"} for _ in frames]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # Get text inputs only
            text_inputs = self.processor(text=prompt_text, return_tensors="pt").to(self.device)
            
            inputs = {
                **text_inputs,
                "pixel_values": pixel_values
            }
            
            t1 = time.time()
            preprocessing_latency = (t1 - t0) * 1000
            
            # 2. Inference
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=100)
                
        else:
            # PATH A: CPU/PIL HuggingFace AutoProcessor
            # 1. Preprocessing (GPU -> CPU -> PIL)
            images = [self._frame_to_pil(f) for f in frames]
            
            # Construct HuggingFace messages for SmolVLM
            content = [{"type": "image"} for _ in frames]
            content.append({"type": "text", "text": prompt})
            
            messages = [{"role": "user", "content": content}]
            
            prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            
            inputs = self.processor(text=prompt_text, images=images, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            t1 = time.time()
            preprocessing_latency = (t1 - t0) * 1000
            
            # 2. Inference
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=100)
                
        generated_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        
        t2 = time.time()
        inference_latency = (t2 - t1) * 1000
        
        return VLMResponse(
            text=generated_texts[0],
            latency_ms=inference_latency,
            preprocessing_latency_ms=preprocessing_latency
        )
