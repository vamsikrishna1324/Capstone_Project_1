"""
Sample Benchmarks Runner for Edge Deployment Models.

Runs a fast sample benchmark across all registered vision models and execution
runtimes (PyTorch CPU/MPS, ONNX Runtime, and TensorFlow Lite), displaying
model sizes, parameter counts, latency percentiles, and throughput.

Usage:
    PYTHONPATH=. ./.venv311/bin/python -m benchmarking.sample_benchmarks
"""

import argparse
import sys
from typing import List, Optional, Tuple

import torch

from benchmarking.config import BenchmarkConfig, BenchmarkRuntime
from benchmarking.engine import BenchmarkEngine
from models.model_registry import list_supported_models


def run_sample_benchmarks(
    models: Optional[List[str]] = None,
    warmup_runs: int = 5,
    num_runs: int = 20,
    batch_size: int = 1,
    measure_memory: bool = True,
) -> None:
    """Execute sample benchmarks across models and runtimes.

    Args:
        models: List of model identifiers to profile (defaults to all supported models).
        warmup_runs: Number of warm-up iterations to exclude from timing.
        num_runs: Number of measured inference iterations.
        batch_size: Inference batch size.
        measure_memory: Whether to profile process memory footprint.
    """
    if models is None:
        models = list_supported_models()

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    runtime_configs: List[Tuple[str, str, Optional[str]]] = [
        ("pytorch", "cpu", None),
    ]
    if mps_available:
        runtime_configs.append(("pytorch", "mps", None))
    runtime_configs.extend([
        ("onnx", "cpu", "CPUExecutionProvider"),
        ("tflite", "cpu", None),
    ])

    if measure_memory:
        header = (
            f"{'Model':<20} | {'Runtime':<8} | {'Device':<6} | "
            f"{'Size':<9} | {'Mean ms':<9} | {'FPS':<8} | "
            f"{'Peak RAM':<10} | {'RAM Delta':<10}"
        )
    else:
        header = (
            f"{'Model':<20} | {'Runtime':<8} | {'Device':<6} | "
            f"{'Size (MB)':<10} | {'Mean (ms)':<10} | {'Median (ms)':<11} | "
            f"{'P95 (ms)':<10} | {'FPS':<10}"
        )
    separator = "-" * len(header)

    print("\n" + separator)
    print("Edge Model Benchmark — Sample Benchmarking Suite (Size, Latency, Memory)")
    print(separator)
    print(f"Iterations: {warmup_runs} warmup + {num_runs} measured | Batch Size: {batch_size} | Memory: {measure_memory}")
    print(separator)
    print(header)
    print(separator)

    for model_id in models:
        for rt, dev, prov in runtime_configs:
            try:
                cfg = BenchmarkConfig(
                    model_id=model_id,
                    runtime=rt,
                    device=dev,
                    backend_provider=prov,
                    warmup_runs=warmup_runs,
                    num_runs=num_runs,
                    batch_size=batch_size,
                )
                engine = BenchmarkEngine(cfg)
                res = engine.run_benchmark(
                    measure_model_size=True,
                    measure_latency=True,
                    measure_memory=measure_memory,
                )

                if res.success and res.latency:
                    lat = res.latency
                    size_str = (
                        f"{res.model_size.file_size_mb:.2f} MB"
                        if res.model_size and res.model_size.file_size_mb is not None
                        else "N/A"
                    )
                    if measure_memory and res.memory_footprint:
                        mem = res.memory_footprint
                        peak_str = f"{mem.peak_ram_mb:.1f} MB" if mem.peak_ram_mb else "N/A"
                        delta_str = f"{mem.ram_delta_mb:.1f} MB" if mem.ram_delta_mb else "N/A"
                        print(
                            f"{model_id:<20} | {rt:<8} | {dev:<6} | "
                            f"{size_str:<9} | {lat.mean_ms:<9.2f} | {lat.throughput_fps:<8.1f} | "
                            f"{peak_str:<10} | {delta_str:<10}"
                        )
                    else:
                        print(
                            f"{model_id:<20} | {rt:<8} | {dev:<6} | "
                            f"{size_str:<10} | {lat.mean_ms:<10.2f} | {lat.median_ms:<11.2f} | "
                            f"{lat.p95_ms:<10.2f} | {lat.throughput_fps:<10.2f}"
                        )
                else:
                    err = res.error_message or "Unknown failure"
                    print(f"{model_id:<20} | {rt:<8} | {dev:<6} | FAILED: {err}")
            except Exception as exc:
                print(f"{model_id:<20} | {rt:<8} | {dev:<6} | ERROR: {exc}")

    print(separator + "\n")


def main() -> None:
    """Parse command line arguments and execute sample benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run sample latency, size, and memory benchmarks across edge vision models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of model IDs to benchmark (default: all registered models).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warm-up iterations (default: 5).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of measured iterations (default: 20).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1).",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable memory footprint measurement.",
    )
    args = parser.parse_args()

    run_sample_benchmarks(
        models=args.models,
        warmup_runs=args.warmup,
        num_runs=args.runs,
        batch_size=args.batch_size,
        measure_memory=not args.no_memory,
    )


if __name__ == "__main__":
    main()
