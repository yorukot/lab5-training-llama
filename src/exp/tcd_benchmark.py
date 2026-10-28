"""Eager CUDA direction-kernel timings, not end-to-end model throughput."""

import argparse
import hashlib
import statistics
from pathlib import Path

import torch

from exp.common import atomic_json
from exp.tcd import crossfit_direction
from muon import muon_update


@torch.no_grad()
def benchmark(output, repeats=20):
    if output.exists():
        raise ValueError("Use a fresh output path")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    torch.manual_seed(901)
    torch.set_float32_matmul_precision("highest")
    result = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "repeats": repeats,
        "warmup": 5,
        "compile": False,
        "tcd_dtype": "float32",
        "normuon_dtype": "native BF16 Polar Express + FP32 row equalization",
        "scope": "Direction kernels include averaging; exclude backward, gradient capture, parameter step, and compilation",
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("tcd.py", "tcd_benchmark.py")
        },
        "timings": [],
    }
    for shape in ((128, 512), (512, 512), (2048, 512)):
        samples = torch.randn(4, *shape, device="cuda", dtype=torch.float32)
        state = {}

        def normuon(sample_batch=samples, update_state=state):
            return muon_update(sample_batch.mean(0), update_state, "normuon")

        kernels = {
            "normuon": normuon,
            "tcd_exact": lambda sample_batch=samples: crossfit_direction(
                sample_batch, rho=3, solver="exact"
            ),
            "tcd_first_order": lambda sample_batch=samples: crossfit_direction(
                sample_batch, rho=3, solver="first_order"
            ),
        }
        for name, kernel in kernels.items():
            for _ in range(5):
                value = kernel()
            if not value.isfinite().all():
                raise FloatingPointError(name)
            del value
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            elapsed = []
            for _ in range(repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                value = kernel()
                end.record()
                end.synchronize()
                elapsed.append(start.elapsed_time(end))
                del value
            row = {
                "shape": shape,
                "method": name,
                "median_ms": statistics.median(elapsed),
                "min_ms": min(elapsed),
                "max_ms": max(elapsed),
                "elapsed_ms": elapsed,
                "additional_peak_bytes": torch.cuda.max_memory_allocated() - base,
            }
            result["timings"].append(row)
            print(shape, name, row["median_ms"], flush=True)
            atomic_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark(args.output)
