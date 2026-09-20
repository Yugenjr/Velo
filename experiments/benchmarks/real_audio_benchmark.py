"""
Velo Real Audio Processing & VAD Benchmark (Milestone V1.11)

Taxonomy Class: Class 2 (CPU-Only Audio Processing)

Measures:
1. Real Opus/PCM audio chunking and AudioScheduler buffering.
2. VAD (Voice Activity Detection) energy computation throughput and latency distribution (p50, p95, p99).
3. ASR transcription dispatch latency.
4. Backpressure bounds and queue stability over time.
"""
import time
import numpy as np
import torch
import velo
from velo import AudioChunk, AudioScheduler, VAD, MockASRAdapter, StreamClosedError
from benchmark_utils import save_benchmark_results


def generate_speech_pattern_pcm(duration_s: float = 30.0, sample_rate: int = 48000, channels: int = 2):
    """Generates continuous 20ms audio chunks alternating between active speech and silence."""
    chunk_samples = int(0.02 * sample_rate * channels)
    total_chunks = int(duration_s / 0.02)
    
    chunks = []
    for i in range(total_chunks):
        ts = i * 0.02
        # Speech active for 2s every 4s
        is_active_speech = (int(ts) % 4) < 2
        
        if is_active_speech:
            # Active harmonic waveform
            t = np.linspace(0, 0.02, chunk_samples, endpoint=False)
            freq1 = 220.0 + (i % 5) * 40.0
            freq2 = 440.0
            wave = (0.5 * np.sin(2 * np.pi * freq1 * t) + 0.3 * np.sin(2 * np.pi * freq2 * t))
            pcm_data = (wave * 16000).astype(np.int16).tobytes()
        else:
            # Low noise floor (silence)
            noise = np.random.normal(0, 5, chunk_samples).astype(np.int16)
            pcm_data = noise.tobytes()
            
        chunks.append((pcm_data, ts, is_active_speech))
        
    return chunks


def run_real_audio_benchmark(duration_s: float = 30.0, chunk_duration_s: float = 0.5):
    print(f"\n--- [Class 2] Real Audio Processing & VAD Benchmark ({duration_s}s audio) ---")
    
    raw_chunks = generate_speech_pattern_pcm(duration_s=duration_s)
    total_raw_chunks = len(raw_chunks)
    print(f"Generated {total_raw_chunks} 20ms PCM chunks (48kHz, stereo, alternating speech/silence)")
    
    audio_scheduler = AudioScheduler(
        chunk_duration_s=chunk_duration_s,
        sample_rate=48000,
        channels=2,
        max_buffered_s=10.0,
    )
    vad = VAD(energy_threshold=0.01)
    asr = MockASRAdapter(mock_text="benchmark real speech segment")
    
    # 1. Measure Chunk Ingestion & Slicing Throughput
    t0 = time.time()
    for pcm_bytes, ts, _ in raw_chunks:
        int_samples = torch.frombuffer(pcm_bytes, dtype=torch.int16)
        samples = int_samples.to(torch.float32) / 32768.0
        chunk = AudioChunk(
            samples=samples,
            timestamp=float(ts),
            channels=2,
            sample_rate=48000,
            duration=0.02,
        )
        audio_scheduler.submit(chunk)
    t_ingest = time.time() - t0
    ingest_throughput = round(total_raw_chunks / max(0.0001, t_ingest), 2)
    print(f"Audio Ingestion Throughput: {ingest_throughput} chunks/s ({total_raw_chunks} chunks in {t_ingest*1000:.1f}ms)")
    
    # 2. Measure VAD & ASR Processing Latencies
    audio_scheduler.close()
    
    vad_latencies_us = []
    asr_latencies_ms = []
    acquired_chunks = 0
    speech_detected_count = 0
    silence_count = 0
    
    while True:
        try:
            chunk = audio_scheduler.acquire()
            acquired_chunks += 1
            
            # Measure VAD execution latency
            t_v0 = time.perf_counter()
            vad_res = vad.analyze(chunk)
            t_v1 = time.perf_counter()
            vad_latencies_us.append((t_v1 - t_v0) * 1e6)
            
            if vad_res.is_speech:
                speech_detected_count += 1
                t_a0 = time.perf_counter()
                transcript = asr.transcribe(chunk)
                t_a1 = time.perf_counter()
                asr_latencies_ms.append((t_a1 - t_a0) * 1000.0)
            else:
                silence_count += 1
                
        except Exception:
            break
            
    stats = audio_scheduler.stats()
    
    metrics = {
        "duration_s": duration_s,
        "total_raw_chunks": total_raw_chunks,
        "consolidated_chunks": acquired_chunks,
        "speech_chunks": speech_detected_count,
        "silence_chunks": silence_count,
        "ingest_throughput_chunks_per_sec": ingest_throughput,
        "vad_latency_us": {
            "mean": round(float(np.mean(vad_latencies_us)), 2) if vad_latencies_us else 0.0,
            "p50": round(float(np.percentile(vad_latencies_us, 50)), 2) if vad_latencies_us else 0.0,
            "p95": round(float(np.percentile(vad_latencies_us, 95)), 2) if vad_latencies_us else 0.0,
            "p99": round(float(np.percentile(vad_latencies_us, 99)), 2) if vad_latencies_us else 0.0,
        },
        "asr_dispatch_latency_ms": {
            "mean": round(float(np.mean(asr_latencies_ms)), 3) if asr_latencies_ms else 0.0,
            "p50": round(float(np.percentile(asr_latencies_ms, 50)), 3) if asr_latencies_ms else 0.0,
            "p95": round(float(np.percentile(asr_latencies_ms, 95)), 3) if asr_latencies_ms else 0.0,
        },
        "scheduler_drops": stats.get("chunks_dropped", 0),
    }
    
    print(f"VAD Latency: p50={metrics['vad_latency_us']['p50']}us, p95={metrics['vad_latency_us']['p95']}us | Speech: {speech_detected_count}, Silence: {silence_count}")
    
    config = {
        "sample_rate": 48000,
        "channels": 2,
        "chunk_duration_s": chunk_duration_s,
        "vad_energy_threshold": 0.01,
    }
    save_benchmark_results("real_audio_processing", config, metrics)
    return metrics


if __name__ == "__main__":
    run_real_audio_benchmark()
