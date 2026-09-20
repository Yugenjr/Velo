import os
import sys
import platform
import time
import json
import csv
from typing import Dict, Any, Optional

try:
    import torch
except ImportError:
    torch = None


def get_system_metadata() -> Dict[str, Any]:
    """Retrieve execution environment metadata without hard-coding."""
    gpu_info = {
        "available": False,
        "name": None,
        "count": 0,
        "cuda_version": None,
        "memory_total_mb": 0.0,
    }
    if torch and torch.cuda.is_available():
        gpu_info["available"] = True
        gpu_info["name"] = torch.cuda.get_device_name(0)
        gpu_info["count"] = torch.cuda.device_count()
        gpu_info["cuda_version"] = torch.version.cuda
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            gpu_info["memory_total_mb"] = round(total_mem, 2)
        except Exception:
            pass

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__ if torch else None,
        "gpu": gpu_info,
    }


def save_benchmark_results(name: str, config: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """Save benchmark results to structured JSON."""
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = int(time.time())
    filename = f"benchmark_{name}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)

    output = {
        "benchmark": name,
        "environment": get_system_metadata(),
        "configuration": config,
        "metrics": metrics,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[BENCHMARK] Results saved to: {filepath}")
    return filepath
