#!/usr/bin/env python3
"""
Benchmark script for FSST implementations.

Usage:
    pixi run python scripts/benchmark_fsst.py

On x86 systems, you can also compare against the original MATLAB-generated library:
    pixi run pip install fsst==0.1.1 --extra-index-url https://gitlab.com/api/v4/projects/62793076/packages/pypi/simple
    pixi run python scripts/benchmark_fsst.py --compare-original
"""

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np
import scipy.signal


@dataclass
class BenchmarkResult:
    name: str
    signal_length: int
    iterations: int
    total_time: float
    mean_time: float
    std_time: float
    throughput: float  # samples/second


def benchmark_fsst(fsst_func, signal: np.ndarray, window: np.ndarray, fs: float, iterations: int = 10) -> list[float]:
    """Run benchmark and return list of execution times."""
    times = []

    # Warmup
    _ = fsst_func(signal, fs, window)

    for _ in range(iterations):
        start = time.perf_counter()
        _ = fsst_func(signal, fs, window)
        end = time.perf_counter()
        times.append(end - start)

    return times


def run_benchmarks(
    fsst_func, name: str, signal_lengths: list[int], fs: float, window: np.ndarray, iterations: int = 10
) -> list[BenchmarkResult]:
    """Run benchmarks for different signal lengths."""
    results = []

    for length in signal_lengths:
        # Generate test signal
        t = np.arange(length) / fs
        signal = np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

        times = benchmark_fsst(fsst_func, signal, window, fs, iterations)

        mean_time = np.mean(times)
        std_time = np.std(times)
        throughput = length / mean_time

        result = BenchmarkResult(
            name=name,
            signal_length=length,
            iterations=iterations,
            total_time=sum(times),
            mean_time=mean_time,
            std_time=std_time,
            throughput=throughput,
        )
        results.append(result)

    return results


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a table format."""
    print(f"\n{'='*80}")
    print(f"{'Implementation':<20} {'Samples':<12} {'Mean (ms)':<12} {'Std (ms)':<12} {'Throughput (kS/s)':<18}")
    print(f"{'='*80}")

    for r in results:
        print(f"{r.name:<20} {r.signal_length:<12} {r.mean_time*1000:<12.3f} {r.std_time*1000:<12.3f} {r.throughput/1000:<18.1f}")

    print(f"{'='*80}\n")


def print_comparison(results_new: list[BenchmarkResult], results_old: list[BenchmarkResult]) -> None:
    """Print comparison between two implementations."""
    print(f"\n{'='*90}")
    print(f"{'Samples':<12} {'New (ms)':<12} {'Old (ms)':<12} {'Speedup':<12} {'New kS/s':<15} {'Old kS/s':<15}")
    print(f"{'='*90}")

    for r_new, r_old in zip(results_new, results_old):
        speedup = r_old.mean_time / r_new.mean_time
        print(
            f"{r_new.signal_length:<12} "
            f"{r_new.mean_time*1000:<12.3f} "
            f"{r_old.mean_time*1000:<12.3f} "
            f"{speedup:<12.2f}x "
            f"{r_new.throughput/1000:<15.1f} "
            f"{r_old.throughput/1000:<15.1f}"
        )

    print(f"{'='*90}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark FSST implementations")
    parser.add_argument("--compare-original", action="store_true", help="Compare against original MATLAB-generated library")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations per benchmark")
    parser.add_argument(
        "--signal-lengths",
        type=int,
        nargs="+",
        default=[1000, 2000, 5000, 10000, 20000, 50000],
        help="Signal lengths to benchmark",
    )
    args = parser.parse_args()

    # Configuration matching main.py
    fs = 1000.0
    window = scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False)

    print("\nFSST Benchmark")
    print("=" * 40)
    print(f"Sample rate: {fs} Hz")
    print(f"Window: Kaiser (beta=0.5), length={len(window)}")
    print(f"Iterations per test: {args.iterations}")
    print(f"Signal lengths: {args.signal_lengths}")

    # Import new implementation
    try:
        import ssq

        print("\n[OK] New C++ FSST implementation loaded")
    except ImportError as e:
        print(f"\n[ERROR] Failed to import ssq: {e}")
        print("Make sure ssq is installed")
        sys.exit(1)

    # Benchmark new implementation
    print("\nBenchmarking new C++ implementation...")
    results_new = run_benchmarks(ssq.fsst, "C++ (FFTW)", args.signal_lengths, fs, window, args.iterations)
    print_results(results_new)

    # Try to import and benchmark original implementation
    if args.compare_original:
        try:
            # Try to import original (will fail on ARM64)
            from fsst import fsst as fsst_old

            print("[OK] Original MATLAB-generated FSST implementation loaded")

            print("\nBenchmarking original MATLAB-generated implementation...")
            results_old = run_benchmarks(fsst_old, "MATLAB (Original)", args.signal_lengths, fs, window, args.iterations)
            print_results(results_old)

            print("\nComparison (New vs Original):")
            print_comparison(results_new, results_old)

        except ImportError as e:
            print(f"\n[WARNING] Original fsst library not available: {e}")
            print("On x86 systems, install with:")
            print("  pixi run pip install fsst==0.1.1 --extra-index-url https://gitlab.com/api/v4/projects/62793076/packages/pypi/simple")

    # Summary
    print("\nSummary:")
    print("-" * 40)
    avg_throughput = np.mean([r.throughput for r in results_new])
    print(f"Average throughput: {avg_throughput/1000:.1f} kS/s")
    print(f"For 2000-sample signal: {results_new[1].mean_time*1000:.2f} ms" if len(results_new) > 1 else "")


if __name__ == "__main__":
    main()
