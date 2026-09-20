"""
Velo Milestone V1.11 Master Benchmark Suite

Executes all Real Hardware, Real Media, and Real Pipeline benchmarks with:
1. 3-run repeatability (Run 1, Run 2, Run 3, Mean, StdDev).
2. Explicit isolation of warmup vs steady state.
3. Environment metadata recording.
4. Structured JSON/CSV persistence and human-readable summary reporting.
"""
import os
import sys
import time
import json
import numpy as np
import torch
import velo
from benchmark_utils import save_benchmark_results, get_system_metadata
from real_video_benchmark import run_resolution_benchmark
from real_audio_benchmark import run_real_audio_benchmark
from real_sync_benchmark import run_real_sync_benchmark
from real_multimodal_benchmark import run_multimodal_pipeline_benchmark, evaluate_configuration_b
import asyncio


def calculate_repeatability_stats(runs_data):
    """Calculates Mean and StdDev across multiple runs for numerical values."""
    if not runs_data:
        return {}
    
    first = runs_data[0]
    if isinstance(first, (int, float)):
        vals = [float(r) for r in runs_data]
        return {
            "runs": vals,
            "mean": round(float(np.mean(vals)), 3),
            "std": round(float(np.std(vals)), 3),
        }
    elif isinstance(first, dict):
        result = {}
        for k in first:
            sub_runs = [r[k] for r in runs_data if k in r]
            result[k] = calculate_repeatability_stats(sub_runs)
        return result
    return runs_data[-1]


def run_full_v1_11_benchmark_suite(num_iterations: int = 3):
    print("================================================================================")
    print(f"VELO V1.11 REAL HARDWARE & MEDIA BENCHMARK SUITE ({num_iterations} REPEATED ITERATIONS)")
    print("================================================================================")
    
    env_meta = get_system_metadata()
    print(f"Platform: {env_meta['os']} {env_meta['os_release']} | Python: {env_meta['python_version']} | PyTorch: {env_meta['torch_version']}")
    gpu = env_meta["gpu"]
    if gpu["available"]:
        print(f"GPU: {gpu['name']} | CUDA: {gpu['cuda_version']} | VRAM: {gpu['memory_total_mb']} MB")
    else:
        print("GPU: None (CPU Mode)")
    print("================================================================================\n")
    
    suite_results = {
        "environment": env_meta,
        "taxonomy_breakdown": {
            "real_h264_nvdec": "Class 4 (Real Media & Hardware NVDEC)",
            "real_audio_processing": "Class 2 (CPU-Only Audio Processing)",
            "real_sync": "Class 4+2 (Real NVDEC Video + Real Audio Correlation)",
            "real_multimodal_config_a": "Class 4+2+6A (Real NVDEC + Real Audio + Mock Adapter)",
            "real_multimodal_config_b": "Class 4+2+6B (Real SmolVLM, VRAM Dependent)",
        },
        "benchmarks": {},
    }
    
    # ---------------------------------------------------------
    # 1. REAL H.264 HARDWARE NVDEC BENCHMARK (3 RUNS)
    # ---------------------------------------------------------
    print("\n>>> BENCHMARK 1: Real H.264 Hardware NVDEC Throughput (3 Runs) <<<")
    resolutions = [
        (640, 480, "480p", 60),
        (1280, 720, "720p", 60),
        (1920, 1080, "1080p", 60),
        (3840, 2160, "4k", 30),
    ]
    
    video_suite = {}
    for w, h, name, num_f in resolutions:
        runs = []
        for i in range(num_iterations):
            print(f"  [Run {i+1}/{num_iterations}] Benchmarking {name} ({w}x{h})...")
            res = run_resolution_benchmark(w, h, name, num_frames=num_f)
            runs.append(res)
            time.sleep(0.5)
            
        fps_values = [r["max_nvdec_throughput_fps"] for r in runs]
        dlpack_us = [r["dlpack_conversion_us"]["p50"] for r in runs]
        intervals_p50 = [r["frame_interval_ms"]["p50"] for r in runs]
        
        video_suite[name] = {
            "width": w,
            "height": h,
            "nvdec_fps": {
                "runs": fps_values,
                "mean": round(float(np.mean(fps_values)), 2),
                "std": round(float(np.std(fps_values)), 2),
            },
            "frame_interval_p50_ms": {
                "runs": intervals_p50,
                "mean": round(float(np.mean(intervals_p50)), 3),
                "std": round(float(np.std(intervals_p50)), 3),
            },
            "dlpack_zero_copy_overhead_p50_us": {
                "runs": dlpack_us,
                "mean": round(float(np.mean(dlpack_us)), 2),
                "std": round(float(np.std(dlpack_us)), 2),
            },
            "gpu_memory_allocated_mb": runs[-1]["gpu_memory_allocated_mb"],
        }
        
    suite_results["benchmarks"]["real_h264_nvdec"] = video_suite

    # ---------------------------------------------------------
    # 2. REAL AUDIO PROCESSING & VAD (3 RUNS)
    # ---------------------------------------------------------
    print("\n>>> BENCHMARK 2: Real Audio Processing & VAD Latency (3 Runs) <<<")
    audio_runs = []
    for i in range(num_iterations):
        print(f"  [Run {i+1}/{num_iterations}] Running Audio Benchmark...")
        res = run_real_audio_benchmark(duration_s=15.0, chunk_duration_s=0.5)
        audio_runs.append(res)
        time.sleep(0.5)
        
    audio_ingest_rates = [r["ingest_throughput_chunks_per_sec"] for r in audio_runs]
    vad_p50 = [r["vad_latency_us"]["p50"] for r in audio_runs]
    vad_p95 = [r["vad_latency_us"]["p95"] for r in audio_runs]
    
    suite_results["benchmarks"]["real_audio_processing"] = {
        "ingest_throughput_chunks_per_sec": {
            "runs": audio_ingest_rates,
            "mean": round(float(np.mean(audio_ingest_rates)), 2),
            "std": round(float(np.std(audio_ingest_rates)), 2),
        },
        "vad_latency_p50_us": {
            "runs": vad_p50,
            "mean": round(float(np.mean(vad_p50)), 2),
            "std": round(float(np.std(vad_p50)), 2),
        },
        "vad_latency_p95_us": {
            "runs": vad_p95,
            "mean": round(float(np.mean(vad_p95)), 2),
            "std": round(float(np.std(vad_p95)), 2),
        },
    }

    # ---------------------------------------------------------
    # 3. REAL AUDIO/VIDEO SYNCHRONIZATION (3 RUNS)
    # ---------------------------------------------------------
    print("\n>>> BENCHMARK 3: Real Audio/Video Synchronization (3 Runs) <<<")
    sync_runs = []
    for i in range(num_iterations):
        print(f"  [Run {i+1}/{num_iterations}] Running Sync Benchmark...")
        res = run_real_sync_benchmark(num_frames=60)
        sync_runs.append(res)
        time.sleep(0.5)
        
    skew_means = [r["timestamp_skew_ms"]["mean"] for r in sync_runs]
    skew_p95 = [r["timestamp_skew_ms"]["p95"] for r in sync_runs]
    skew_max = [r["timestamp_skew_ms"]["max"] for r in sync_runs]
    lookup_p50 = [r["context_lookup_latency_us"]["p50"] for r in sync_runs]
    
    suite_results["benchmarks"]["real_av_synchronization"] = {
        "timestamp_skew_mean_ms": {
            "runs": skew_means,
            "mean": round(float(np.mean(skew_means)), 3),
            "std": round(float(np.std(skew_means)), 3),
        },
        "timestamp_skew_p95_ms": {
            "runs": skew_p95,
            "mean": round(float(np.mean(skew_p95)), 3),
            "std": round(float(np.std(skew_p95)), 3),
        },
        "timestamp_skew_max_ms": {
            "runs": skew_max,
            "mean": round(float(np.mean(skew_max)), 3),
            "std": round(float(np.std(skew_max)), 3),
        },
        "context_lookup_p50_us": {
            "runs": lookup_p50,
            "mean": round(float(np.mean(lookup_p50)), 2),
            "std": round(float(np.std(lookup_p50)), 2),
        },
    }

    # ---------------------------------------------------------
    # 4. REAL MULTIMODAL PIPELINE (3 RUNS)
    # ---------------------------------------------------------
    print("\n>>> BENCHMARK 4: Real Multimodal Pipeline End-to-End (3 Runs) <<<")
    pipeline_runs = []
    for i in range(num_iterations):
        print(f"  [Run {i+1}/{num_iterations}] Running Pipeline Benchmark...")
        res = asyncio.run(run_multimodal_pipeline_benchmark(duration_s=5.0, target_fps=15.0, simulated_model_latency_ms=10.0))
        pipeline_runs.append(res)
        time.sleep(0.5)
        
    e2e_fps = [r["effective_fps"] for r in pipeline_runs]
    inf_p50 = [r["inference_latency_ms"]["p50"] for r in pipeline_runs]
    e2e_p50 = [r["end_to_end_latency_ms"]["p50"] for r in pipeline_runs]
    e2e_p95 = [r["end_to_end_latency_ms"]["p95"] for r in pipeline_runs]
    
    suite_results["benchmarks"]["real_multimodal_pipeline_config_a"] = {
        "effective_fps": {
            "runs": e2e_fps,
            "mean": round(float(np.mean(e2e_fps)), 2),
            "std": round(float(np.std(e2e_fps)), 2),
        },
        "inference_latency_p50_ms": {
            "runs": inf_p50,
            "mean": round(float(np.mean(inf_p50)), 3),
            "std": round(float(np.std(inf_p50)), 3),
        },
        "end_to_end_latency_p50_ms": {
            "runs": e2e_p50,
            "mean": round(float(np.mean(e2e_p50)), 3),
            "std": round(float(np.std(e2e_p50)), 3),
        },
        "end_to_end_latency_p95_ms": {
            "runs": e2e_p95,
            "mean": round(float(np.mean(e2e_p95)), 3),
            "std": round(float(np.std(e2e_p95)), 3),
        },
    }
    
    # Evaluate Config B (SmolVLM)
    suite_results["benchmarks"]["real_multimodal_pipeline_config_b"] = evaluate_configuration_b()

    # Save Master Suite JSON
    filepath = save_benchmark_results("v1_11_comprehensive_suite", {"iterations": num_iterations}, suite_results)
    
    # ---------------------------------------------------------
    # PRINT HUMAN READABLE SUMMARY TABLE
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("                     VELO V1.11 BENCHMARK SUMMARY TABLE")
    print("="*80)
    print(f"{'Benchmark Domain':<30} | {'Scenario':<18} | {'Mean (u)':<14} | {'StdDev (o)':<10}")
    print("-"*80)
    
    # Video NVDEC
    for name, v in video_suite.items():
        print(f"{'NVDEC Decode FPS (' + name + ')':<30} | {'Unthrottled':<18} | {str(v['nvdec_fps']['mean']) + ' FPS':<14} | {str(v['nvdec_fps']['std']):<10}")
        print(f"{'DLPack Zero-Copy (' + name + ')':<30} | {'to_torch()':<18} | {str(v['dlpack_zero_copy_overhead_p50_us']['mean']) + ' us':<14} | {str(v['dlpack_zero_copy_overhead_p50_us']['std']):<10}")
        
    print("-"*80)
    # Audio
    a = suite_results["benchmarks"]["real_audio_processing"]
    print(f"{'Audio Ingest Throughput':<30} | {'48kHz 2ch':<18} | {str(a['ingest_throughput_chunks_per_sec']['mean']) + ' ch/s':<14} | {str(a['ingest_throughput_chunks_per_sec']['std']):<10}")
    print(f"{'VAD Analysis Latency':<30} | {'p50':<18} | {str(a['vad_latency_p50_us']['mean']) + ' us':<14} | {str(a['vad_latency_p50_us']['std']):<10}")
    
    print("-"*80)
    # Sync
    s = suite_results["benchmarks"]["real_av_synchronization"]
    print(f"{'Audio/Video Skew':<30} | {'Mean':<18} | {str(s['timestamp_skew_mean_ms']['mean']) + ' ms':<14} | {str(s['timestamp_skew_mean_ms']['std']):<10}")
    print(f"{'Audio/Video Skew':<30} | {'p95':<18} | {str(s['timestamp_skew_p95_ms']['mean']) + ' ms':<14} | {str(s['timestamp_skew_p95_ms']['std']):<10}")
    print(f"{'Temporal Context Lookup':<30} | {'p50':<18} | {str(s['context_lookup_p50_us']['mean']) + ' us':<14} | {str(s['context_lookup_p50_us']['std']):<10}")
    
    print("-"*80)
    # Pipeline
    p = suite_results["benchmarks"]["real_multimodal_pipeline_config_a"]
    print(f"{'Multimodal Pipeline Rate':<30} | {'Effective FPS':<18} | {str(p['effective_fps']['mean']) + ' FPS':<14} | {str(p['effective_fps']['std']):<10}")
    print(f"{'End-to-End Latency':<30} | {'p50':<18} | {str(p['end_to_end_latency_p50_ms']['mean']) + ' ms':<14} | {str(p['end_to_end_latency_p50_ms']['std']):<10}")
    print(f"{'End-to-End Latency':<30} | {'p95':<18} | {str(p['end_to_end_latency_p95_ms']['mean']) + ' ms':<14} | {str(p['end_to_end_latency_p95_ms']['std']):<10}")
    
    print("="*80)
    print(f"Configuration B (Real SmolVLM): {suite_results['benchmarks']['real_multimodal_pipeline_config_b']['status']}")
    print("================================================================================\n")
    return suite_results


if __name__ == "__main__":
    run_full_v1_11_benchmark_suite(num_iterations=3)
