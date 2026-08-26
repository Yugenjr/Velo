import time
import torch
import gc
from velo.preprocess import GPUPreprocessor

class MockFrame:
    def __init__(self, shape=(1080, 1920, 4)):
        self.shape = shape
        self._tensor = torch.randint(0, 255, shape, dtype=torch.uint8)
        if torch.cuda.is_available():
            self._tensor = self._tensor.to("cuda")
            
    def to_torch(self):
        return self._tensor

def benchmark_preprocessing():
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return
        
    print("=" * 60)
    print("      VELO PREPROCESSING BENCHMARK (Synthetic)    ")
    print("=" * 60)
    
    preprocessor = GPUPreprocessor(target_size=(384, 384), dtype=torch.bfloat16)
    
    # Warmup
    frame = MockFrame()
    preprocessor.preprocess(frame)
    tensor = frame.to_torch()
    array = tensor.cpu().numpy()
    
    iterations = 100
    
    # PATH A: CPU / NumPy baseline
    torch.cuda.synchronize()
    t0 = time.time()
    
    for _ in range(iterations):
        t = frame.to_torch()
        # 1. H2D -> D2H Transfer
        t_cpu = t.cpu()
        # 2. NumPy conversion
        arr = t_cpu.numpy()
        # 3. Simulate minimal PIL Image/Resize overhead on CPU
        # (Omitted actual PIL resize to be generous to CPU path)
        
    torch.cuda.synchronize()
    cpu_latency = (time.time() - t0) * 1000 / iterations
    
    # PATH B: GPU Zero-Copy Preprocessing
    torch.cuda.synchronize()
    t0 = time.time()
    
    for _ in range(iterations):
        out = preprocessor.preprocess(frame)
        
    torch.cuda.synchronize()
    gpu_latency = (time.time() - t0) * 1000 / iterations
    
    print(f"Path A (CPU/NumPy Extraction only): {cpu_latency:.2f} ms / frame")
    print(f"Path B (GPU Full Preprocessing)   : {gpu_latency:.2f} ms / frame")
    print("=" * 60)
    if gpu_latency < cpu_latency:
        speedup = cpu_latency / gpu_latency
        print(f"GPU Preprocessing is {speedup:.1f}x faster!")

if __name__ == "__main__":
    benchmark_preprocessing()
