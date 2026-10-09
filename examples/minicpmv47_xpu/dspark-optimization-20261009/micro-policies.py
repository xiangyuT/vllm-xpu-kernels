"""Validate full-vocab policies and compare three steady graph blocks."""

import json
import statistics
import time
from pathlib import Path

import torch

ROOT = Path("/workspace/evidence")
torch.ops.load_library(
    "/workspace/preparation/dspark-optimization-20261009/markov-tuning.so"
)
rows = []
for device in range(2):
    torch.xpu.set_device(device)
    torch.manual_seed(7)
    torch.xpu.manual_seed_all(7)
    a = torch.randn((1, 256), device=f"xpu:{device}", dtype=torch.bfloat16)
    w = torch.randn((248192, 256), device=f"xpu:{device}", dtype=torch.bfloat16) * 0.02
    scale = w.float().abs().amax().reshape(1) / 448
    q = (w.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    reference = (a.float() @ (q.float() * scale).t()).to(torch.bfloat16)
    for cfg in range(5):
        torch.ops._minicpm_markov_C.set_config(cfg)
        fn = lambda x=a, v=q, s=scale: torch.ops._minicpm_markov_C.gemv(x, v, s)
        actual = fn()
        torch.testing.assert_close(actual, reference, atol=2e-3, rtol=2e-2)
        for _ in range(4):
            fn()
        torch.xpu.synchronize(device)
        stream = torch.xpu.Stream(device=device)
        stream.wait_stream(torch.xpu.current_stream(device))
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g, stream=stream):
            for _ in range(40):
                out = fn()
        g.replay()
        torch.xpu.synchronize(device)
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            g.replay()
            torch.xpu.synchronize(device)
            samples.append((time.perf_counter() - start) * 1e6 / 40)
        torch.testing.assert_close(out, reference, atol=2e-3, rtol=2e-2)
        row = {
            "logical_device": device,
            "config": cfg,
            "mnk": [1, 248192, 256],
            "samples_us": samples,
            "median_us": statistics.median(samples),
            "seed": 7,
            "calls_per_sample": 40,
            "warmup_calls": 4,
            "numeric_reference": "BF16-rounded FP32 product of actual dequantized FP8 weights",
            "numeric_checked_before_and_after_graph_replay": True,
        }
        rows.append(row)
        print(
            "device",
            device,
            "config",
            cfg,
            "us",
            round(row["median_us"], 3),
            flush=True,
        )
        (ROOT / "micro-policies.json").write_text(json.dumps(rows, indent=2) + "\n")
    del a, w, q, scale, reference, actual, out, g
    torch.xpu.empty_cache()
