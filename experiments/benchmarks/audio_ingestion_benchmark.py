"""
Velo Audio Ingestion & Scheduling Benchmark

Benchmarks audio chunk ingestion throughput, AudioScheduler chunking,
and VAD processing rate.
"""
import time
import sys
import os
import torch
import velo
from velo import AudioChunk, AudioScheduler, VAD, StreamClosedError
from benchmark_utils import save_benchmark_results


class SyntheticAudioSource:
    def __init__(self, sample_rate=48000, channels=2, chunk_ms=20, duration_s=10.0):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_ms = chunk_ms
        self.chunk_duration = chunk_ms / 1000.0
        self.total_chunks = int(duration_s / self.chunk_duration)
        self.chunk_idx = 0

        num_samples = int(self.chunk_duration * sample_rate * channels)
        pcm = (torch.sin(torch.linspace(0, 100, num_samples)) * 16000).to(torch.int16)
        self._raw_bytes = pcm.numpy().tobytes()

    def next_audio(self):
        if self.chunk_idx >= self.total_chunks:
            raise StreamClosedError("End of benchmark audio stream")
        ts = self.chunk_idx * self.chunk_duration
        self.chunk_idx += 1
        return self._raw_bytes, ts, self.channels, self.sample_rate, self.chunk_duration

    def close(self):
        pass


def run_audio_benchmark(duration_s=10.0, chunk_duration_s=0.5):
    source = SyntheticAudioSource(duration_s=duration_s)
    stream = velo.Stream(source)
    audio_scheduler = AudioScheduler(
        chunk_duration_s=chunk_duration_s,
        sample_rate=48000,
        channels=2,
        max_buffered_s=10.0,
    )
    vad = VAD(energy_threshold=0.01)

    print(f"\n--- Running Audio Ingestion Benchmark ({duration_s}s, {source.total_chunks} 20ms chunks) ---")
    t0 = time.time()
    chunks_submitted = 0
    chunks_acquired = 0
    vad_speech_count = 0

    # Ingestion loop
    while True:
        try:
            chunk = stream.next_audio()
            audio_scheduler.submit(chunk)
            chunks_submitted += 1
        except StreamClosedError:
            break

    t_ingest_done = time.time()
    ingest_elapsed = t_ingest_done - t0

    # Drain and VAD loop
    while True:
        try:
            acquired = audio_scheduler.acquire()
            chunks_acquired += 1
            res = vad.analyze(acquired)
            if res.is_speech:
                vad_speech_count += 1
        except Exception:
            break
        if audio_scheduler._buffered_samples < audio_scheduler.target_samples:
            break

    t_total_done = time.time()
    total_elapsed = t_total_done - t0

    stats = audio_scheduler.stats()
    metrics = {
        "duration_s": round(duration_s, 2),
        "total_elapsed_s": round(total_elapsed, 4),
        "raw_chunks_submitted": chunks_submitted,
        "consolidated_chunks_acquired": chunks_acquired,
        "vad_speech_detected_chunks": vad_speech_count,
        "ingestion_chunk_rate_cps": round(chunks_submitted / max(0.001, ingest_elapsed), 2),
        "chunks_dropped": stats.get("chunks_dropped", 0),
        "buffered_duration_s": stats.get("buffered_duration", 0.0),
    }

    print(f"Submitted: {chunks_submitted} raw chunks in {ingest_elapsed:.3f}s ({metrics['ingestion_chunk_rate_cps']} chunks/s)")
    print(f"Acquired: {chunks_acquired} consolidated ({chunk_duration_s}s) chunks | Drops: {metrics['chunks_dropped']}")

    config = {
        "sample_rate": 48000,
        "channels": 2,
        "chunk_duration_s": chunk_duration_s,
        "duration_s": duration_s,
    }
    save_benchmark_results("audio_ingestion", config, metrics)
    return metrics


if __name__ == "__main__":
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    run_audio_benchmark(duration_s=dur)
