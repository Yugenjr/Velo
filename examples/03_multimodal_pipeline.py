"""
Example 03: End-to-End Multimodal Pipeline Orchestration

Demonstrates orchestrating synchronized video + audio streams into
a unified AIPipeline with speech transcription and multimodal AI reasoning.
"""
import asyncio
import velo
import torch

async def main():
    torch.cuda.init()
    
    # 1. Configure pipeline components
    scheduler = velo.AIScheduler(target_fps=5.0, scene_aware=True)
    audio_scheduler = velo.AudioScheduler(chunk_duration_s=0.5, sample_rate=48000, channels=2)
    fusion = velo.TemporalFusion(max_history_s=30.0)
    vad = velo.VAD(energy_threshold=0.01)
    asr = velo.MockASRAdapter(mock_text="User speech command")
    vlm = velo.MockMultimodalAdapter(simulated_latency=0.02)
    
    # 2. In a real application, initialize with live stream:
    # stream = velo.connect_livekit("ws://localhost:7880", token="<TOKEN>")
    # pipeline = velo.AIPipeline(
    #     stream=stream,
    #     scheduler=scheduler,
    #     vlm=vlm,
    #     audio_scheduler=audio_scheduler,
    #     fusion=fusion,
    #     vad=vad,
    #     asr=asr,
    #     context_window_s=2.0,
    # )
    # await pipeline.start()
    # async for response in pipeline.run_inference("Describe what is happening"):
    #     print(f"AI Response: {response.text}")
    # await pipeline.stop()
    print("Velo AIPipeline configured for multimodal video + audio inference.")

if __name__ == "__main__":
    asyncio.run(main())
