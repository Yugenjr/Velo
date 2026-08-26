# Velo VLM Integration

## Why a VLM Abstraction?

Velo owns real-time media ingestion (WebRTC), GPU memory management (NVDEC), and temporal frame scheduling (`AIScheduler`). However, Velo is not a monolithic video analytics framework like NVIDIA DeepStream. 

The `BaseVLMAdapter` abstraction decouples the Velo media ingestion lifecycle from model-specific inference logic, preprocessing, and text generation. This ensures that:
1. **The Scheduler remains model-agnostic**. It only manages pacing and backpressure.
2. **The VLM Adapter strictly owns model logic**. It receives GPU-resident `Frame` objects and returns text/latency.

## The Data Flow

```text
WebRTC
  ↓
NVDEC
  ↓
CUDA Frame (GPU)
  ↓
AIScheduler
  ↓
VLM Adapter
  ↓
Vision-Language Model (e.g. LLaVA, Qwen-VL, SmolVLM)
  ↓
Realtime AI Response
```

## Inference Modes

The adapter natively supports two modes of inference based on the `AIScheduler`'s capabilities.

### Single-Frame Inference
When the AI model just needs to evaluate the latest frame:
```python
frame = scheduler.acquire()
try:
    response = vlm.generate(frame, prompt="Describe this image.")
finally:
    scheduler.release()
```

### Temporal-Context Inference
When the AI model detects an event and requires historical context:
```python
frames = scheduler.snapshot(duration=1.5, max_frames=8)
response = vlm.generate(frames, prompt="Explain the sequence of events.")
```

## Preprocessing Data Paths

Velo supports two data paths for preprocessing depending on the model's capabilities and implementation.

### Path A: Current Hugging Face Processor (CPU-Bound)
Standard `AutoProcessor` pipelines in HuggingFace typically require a PIL Image or Numpy array on the CPU. Therefore, `SmolVLMAdapter` explicitly pulls the CUDA tensor to the CPU via `.cpu().numpy()` and converts it to a PIL image.

```text
NVDEC
 ↓
CUDA Frame
 ↓
CPU transfer (tensor.cpu().numpy())
 ↓
PIL/NumPy
 ↓
HF Processor
 ↓
GPU
 ↓
SmolVLM
```

### Path B: GPU-Native Preprocessing (Zero-Copy)
Velo introduces `GPUPreprocessor`, which transforms the `Frame` into a model-ready PyTorch tensor strictly on the GPU (resizing, scaling, and normalizing). 
```text
NVDEC
 ↓
CUDA Frame
 ↓
GPU preprocessing (GPUPreprocessor)
 ↓
GPU model input (pixel_values)
 ↓
SmolVLM
```

*Note: While Path B eliminates the host-to-device transfers for the vision payload, integrating it flawlessly into complex Hugging Face chat templates (which dynamically inject `<image>` tokens based on the processor output) requires manually bridging the tokenized text and the preprocessed pixel values. Velo's `SmolVLMAdapter(use_gpu_preprocess=True)` achieves this experimentally.*

## Model Requirements

Velo targets high-framerate real-time analysis. When selecting a VLM:
- **VRAM Constraints**: Small models (like the 2B parameter `SmolVLM`) require at least 5-6 GB of VRAM even at `bfloat16`. 
- For local edge deployment, ensure the model fits well within the GPU memory alongside the NVDEC allocation.
