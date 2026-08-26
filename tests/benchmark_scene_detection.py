import time
import torch
from velo.scene import SceneChangeDetector

class MockFrame:
    def __init__(self, shape=(1080, 1920, 4), val=0):
        self.shape = shape
        self._tensor = torch.full(shape, val, dtype=torch.uint8)
        if torch.cuda.is_available():
            self._tensor = self._tensor.to("cuda")

    def to_torch(self):
        return self._tensor

def benchmark_scene_detection():
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return
        
    print("=" * 60)
    print("    VELO SCENE-AWARE SCHEDULING BENCHMARK (Synthetic)    ")
    print("=" * 60)
    
    detector = SceneChangeDetector(threshold=0.05)
    
    # Warmup
    f_warm = MockFrame()
    detector.update(f_warm)
    
    iterations = 500
    
    # Generate a sequence of frames:
    # 80% static (identical to previous)
    # 20% changed
    
    current_val = 100
    
    torch.cuda.synchronize()
    t0 = time.time()
    
    changed_count = 0
    for i in range(iterations):
        if i % 5 == 0:
            current_val = (current_val + 50) % 255 # large change
            
        frame = MockFrame(val=current_val)
        res = detector.update(frame)
        if res.changed:
            changed_count += 1
            
    torch.cuda.synchronize()
    latency = (time.time() - t0) * 1000 / iterations
    
    skipped_count = iterations - changed_count
    
    print(f"Total Frames Analyzed       : {iterations}")
    print(f"Frames Classified Changed   : {changed_count}")
    print(f"Frames Skipped (Unchanged)  : {skipped_count}")
    print(f"Average Detector Latency    : {latency:.3f} ms / frame")
    
    print("\nVLM Inference Savings:")
    print(f"Estimated Inference Reduced : {(skipped_count / iterations)*100:.1f}%")
    print("Real VLM Latency Savings    : BLOCKED / UNMEASURED (RTX 3050 VRAM limitation)")
    print("=" * 60)

if __name__ == "__main__":
    benchmark_scene_detection()
