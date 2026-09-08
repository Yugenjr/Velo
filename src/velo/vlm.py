import time
from typing import Union, List, Optional, Any
from .core import Frame


class VLMResponse:
    """Standardized response from a VLM / multimodal adapter."""
    def __init__(
        self,
        text: str,
        latency_ms: float = 0.0,
        preprocessing_latency_ms: float = 0.0,
        timestamp: Optional[float] = None,
        transcripts: Optional[List[Any]] = None,
        context: Optional[Any] = None,
    ):
        self.text = text
        self.latency_ms = latency_ms
        self.preprocessing_latency_ms = preprocessing_latency_ms
        self.timestamp = timestamp
        self.transcripts = transcripts if transcripts is not None else []
        self.context = context


class BaseVLMAdapter:
    """
    Abstract base class for VLM adapters.
    
    Decouples the specific multimodal AI model implementation from Velo's
    GPU-native scheduling pipeline.
    """
    def generate(
        self,
        frames: Union[Frame, List[Frame]],
        prompt: str,
        context: Optional[Any] = None,
        transcripts: Optional[List[Any]] = None,
    ) -> VLMResponse:
        """
        Run inference on a single frame or a temporal sequence of frames.
        
        Args:
            frames: A single velo.Frame or a list of velo.Frame objects.
            prompt: The text prompt describing what the model should look for.
            context: Optional MultimodalContext containing aligned observations.
            transcripts: Optional list of Transcripts or audio observations.
            
        Returns:
            VLMResponse containing the generated text, metrics, and temporal context.
        """
        raise NotImplementedError


class BaseMultimodalAdapter(BaseVLMAdapter):
    """
    Model-facing abstraction for multimodal inference (video frames + audio/transcripts).
    """
    def generate(
        self,
        frames: Union[Frame, List[Frame]],
        prompt: str,
        context: Optional[Any] = None,
        transcripts: Optional[List[Any]] = None,
    ) -> VLMResponse:
        ts = None
        if isinstance(frames, list) and len(frames) > 0:
            ts = getattr(frames[-1], "timestamp", None)
        elif not isinstance(frames, list):
            ts = getattr(frames, "timestamp", None)
            
        return self.infer(
            frames=frames,
            transcripts=transcripts,
            timestamp=ts,
            prompt=prompt,
            context=context,
        )

    def infer(
        self,
        frames: Union[Frame, List[Frame]],
        transcripts: Optional[List[Any]] = None,
        timestamp: Optional[float] = None,
        prompt: str = "",
        context: Optional[Any] = None,
    ) -> VLMResponse:
        """
        Run multimodal inference on synchronized video frames and audio/transcripts.
        
        Args:
            frames: Single or sequence of GPU-resident Frame objects.
            transcripts: List of Transcript objects overlapping with the temporal window.
            timestamp: Target reference timestamp for correlation.
            prompt: Text prompt / instruction.
            context: Full MultimodalContext object if available.
            
        Returns:
            VLMResponse with generated text and timing metrics.
        """
        raise NotImplementedError


class MockVLM(BaseVLMAdapter):
    """
    A lightweight mock VLM for unit testing.
    Simulates inference latency without downloading models or requiring heavy VRAM.
    """
    def __init__(self, simulated_latency: float = 0.1):
        self.simulated_latency = simulated_latency
        
    def generate(
        self,
        frames: Union[Frame, List[Frame]],
        prompt: str,
        context: Optional[Any] = None,
        transcripts: Optional[List[Any]] = None,
    ) -> VLMResponse:
        t0 = time.time()
        
        if isinstance(frames, list):
            num_frames = len(frames)
            ts = getattr(frames[-1], "timestamp", None) if frames else None
        else:
            num_frames = 1
            ts = getattr(frames, "timestamp", None)
            
        if self.simulated_latency > 0:
            time.sleep(self.simulated_latency)
        
        text = f"Mock response for {num_frames} frames with prompt: '{prompt}'"
        
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000
        
        return VLMResponse(
            text=text,
            latency_ms=latency_ms,
            preprocessing_latency_ms=0.0,
            timestamp=ts,
            transcripts=transcripts,
            context=context,
        )


class MockMultimodalAdapter(BaseMultimodalAdapter):
    """
    A mock multimodal adapter for testing video + audio/transcript integration.
    """
    def __init__(self, simulated_latency: float = 0.05):
        self.simulated_latency = simulated_latency

    def infer(
        self,
        frames: Union[Frame, List[Frame]],
        transcripts: Optional[List[Any]] = None,
        timestamp: Optional[float] = None,
        prompt: str = "",
        context: Optional[Any] = None,
    ) -> VLMResponse:
        t0 = time.time()
        if isinstance(frames, list):
            num_frames = len(frames)
            if timestamp is None and frames:
                timestamp = getattr(frames[-1], "timestamp", None)
        else:
            num_frames = 1
            if timestamp is None:
                timestamp = getattr(frames, "timestamp", None)

        if self.simulated_latency > 0:
            time.sleep(self.simulated_latency)

        transcript_texts = []
        if transcripts:
            for t in transcripts:
                if hasattr(t, "text"):
                    transcript_texts.append(t.text)
                elif isinstance(t, str):
                    transcript_texts.append(t)
        
        joined_transcript = "; ".join(transcript_texts) if transcript_texts else "None"
        text = f"Multimodal response for {num_frames} frames (ts={timestamp}) with transcript: '{joined_transcript}', prompt: '{prompt}'"
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000

        return VLMResponse(
            text=text,
            latency_ms=latency_ms,
            preprocessing_latency_ms=0.0,
            timestamp=timestamp,
            transcripts=transcripts,
            context=context,
        )


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

    def generate(
        self,
        frames: Union[Frame, List[Frame]],
        prompt: str,
        context: Optional[Any] = None,
        transcripts: Optional[List[Any]] = None,
    ) -> VLMResponse:
        import time
        import torch
        
        t0 = time.time()
        
        if not isinstance(frames, list):
            frames = [frames]
            
        if not frames:
            raise ValueError("At least one frame is required for VLM inference.")
            
        timestamp = getattr(frames[-1], "timestamp", None) if frames else None

        # If transcripts are provided, augment the prompt with temporal speech context
        effective_prompt = prompt
        if transcripts:
            transcript_texts = []
            for t in transcripts:
                if hasattr(t, "text"):
                    transcript_texts.append(t.text)
                elif isinstance(t, str):
                    transcript_texts.append(t)
            if transcript_texts:
                context_speech = "; ".join(transcript_texts)
                effective_prompt = f"[Audio context: '{context_speech}'] {prompt}"

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
            content.append({"type": "text", "text": effective_prompt})
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
            content.append({"type": "text", "text": effective_prompt})
            
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
            preprocessing_latency_ms=preprocessing_latency,
            timestamp=timestamp,
            transcripts=transcripts,
            context=context,
        )
