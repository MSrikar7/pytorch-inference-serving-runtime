"""Reproducible local microbatching comparison; not a network benchmark.

Both configurations use the same saved model, generated request inputs, concurrency,
and per-trial request count. Records median across repeated trials, not a single peak.
"""
import argparse
import asyncio
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from app.server import Engine

async def trial(model_path, inputs, concurrency, batch_size, workers, wait_ms):
    engine = Engine(model_path, workers=workers, batch_size=batch_size,
                    batch_wait_ms=wait_ms, queue_limit=max(512, len(inputs)))
    await engine.start()
    sem = asyncio.Semaphore(concurrency)
    latencies = []
    try:
        async def send(features):
            async with sem:
                t0 = time.perf_counter()
                await engine.infer(features)
                latencies.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        await asyncio.gather(*(send(row) for row in inputs))
        elapsed = time.perf_counter() - t0
        metrics = engine.metrics()
    finally:
        await engine.stop()
    latencies.sort()
    return {
        "throughput_rps": round(len(inputs) / elapsed, 2),
        "p50_ms": round(statistics.median(latencies), 3),
        "p95_ms": round(latencies[int((len(latencies)-1) * 0.95)], 3),
        "p99_ms": round(latencies[int((len(latencies)-1) * 0.99)], 3),
        "batches": metrics["batches"],
        "completed": metrics["completed"],
        "failed": metrics["failed"],
        "rejected": metrics["rejected"],
    }

async def compare(args):
    # Identical deterministic synthetic inputs across both configurations.
    rows = [[float((i * 7 + j * 3) % 19) / 19 for j in range(32)]
            for i in range(args.requests)]
    results = {}
    for name, batch_size in (("unbatched", 1), ("microbatched", args.batch_size)):
        runs = []
        for _ in range(args.trials):
            runs.append(await trial(args.model, rows, args.concurrency,
                                    batch_size, args.workers, args.batch_wait_ms))
        results[name] = {
            "trials": runs,
            "median_throughput_rps": round(statistics.median(r["throughput_rps"] for r in runs), 2),
            "median_p95_ms": round(statistics.median(r["p95_ms"] for r in runs), 3),
        }
    a = results["unbatched"]["median_throughput_rps"]
    b = results["microbatched"]["median_throughput_rps"]
    report = {
        "methodology": "same CPU, model, workload, workers and concurrency; direct in-process Engine comparison; no network",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "configuration": vars(args),
        "results": results,
        "throughput_ratio_microbatched_over_unbatched": round(b / a, 3) if a else None,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/model.pt")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-wait-ms", type=float, default=5)
    parser.add_argument("--output", default="results/comparison.json")
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.trials, args.workers, args.batch_size) < 1:
        parser.error("request, concurrency, trial, worker and batch counts must be positive")
    asyncio.run(compare(args))

if __name__ == "__main__":
    main()
