"""Exact-shape Markov BF16/online-FP8 projection checks and three-block timing."""

import json
import statistics
import time
from pathlib import Path

import torch

ROOT = Path("/workspace/evidence")
results = []
for device in range(2):
    torch.xpu.set_device(device)
    torch.manual_seed(7)
    torch.xpu.manual_seed_all(7)
    a = torch.randn((1, 256), device=f"xpu:{device}", dtype=torch.bfloat16)
    w = torch.randn((248192, 256), device=f"xpu:{device}", dtype=torch.bfloat16) * 0.02
    scale = w.float().abs().amax().reshape(1) / 448
    q = (w.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    expected = (a.float() @ (q.float() * scale).t()).to(torch.bfloat16)
    actual = torch.ops._xpu_C.fp8_gemm_w8a16(a, q.t(), scale, None)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
    for name, fn in [
        ("bf16", lambda x=a, v=w: x @ v.t()),
        (
            "online_fp8_generic",
            lambda x=a, v=q, s=scale: torch.ops._xpu_C.fp8_gemm_w8a16(
                x, v.t(), s, None
            ),
        ),
    ]:
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
        assert torch.isfinite(out).all().item()
        results.append(
            {
                "logical_device": device,
                "name": name,
                "mnk": [1, 248192, 256],
                "samples_us": samples,
                "median_us": statistics.median(samples),
                "warmup_calls": 4,
                "calls_per_block": 40,
                "seed": 7,
                "numerical_reference": "FP32 product of dequantized FP8 weights; BF16 output",
                "numerical_check_passed": True,
            }
        )
        print(device, name, round(statistics.median(samples), 3), flush=True)
    del a, w, q, scale, expected, actual, out, g
    torch.xpu.empty_cache()
(ROOT / "micro-markov-generic.json").write_text(json.dumps(results, indent=2) + "\n")
