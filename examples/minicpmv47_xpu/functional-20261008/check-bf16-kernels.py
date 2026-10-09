import importlib.util
import json
from pathlib import Path
import sys
import traceback

import torch
import vllm_xpu_kernels._xpu_C
import vllm_xpu_kernels._moe_C

root = Path('/workspace/preparation/kernels-bf16')
torch.ops.load_library('/workspace/preparation/functional-20261008/native/bf16-diagnostic.so')
sys.path.insert(0, str(root))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

load('vllm_xpu_kernels.moe_shared_fused_interface', root / 'vllm_xpu_kernels/moe_shared_fused_interface.py')
norm = load('bf16_norm_tests', root / 'tests/test_norm_fp8_gemv.py')
router = load('bf16_router_tests', root / 'tests/fused_moe/test_router_gemv_topk_softmax.py')
moe = load('bf16_moe_tests', root / 'tests/fused_moe/test_moe_shared_fused_decode.py')
gemv = load('bf16_gemv_tests', root / 'tests/test_fp8_gemv_dispatch.py')
results = []

def check(name, function, *args):
    try:
        function(*args)
        torch.xpu.synchronize()
        result = {'test': name, 'passed': True}
    except Exception:
        result = {'test': name, 'passed': False, 'error': traceback.format_exc()}
    results.append(result)
    Path('/workspace/runtime/bf16-kernel-results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(name, 'PASS' if result['passed'] else result['error'], flush=True)

def raw_gemv(dtype, value=None):
    torch.manual_seed(7)
    x = torch.randn(1, 2048, device='xpu', dtype=dtype)
    w = (torch.randn(32, 2048, device='xpu') * .05).to(torch.float8_e4m3fn)
    scale = torch.tensor([.02], device='xpu')
    if value is not None:
        x.zero_(); x[:, 0] = value
        w = torch.zeros(32, 2048, device='xpu'); w[:, 0] = 1
        w = w.to(torch.float8_e4m3fn); scale.fill_(1)
    out = torch.ops._xpu_C.bf16_diagnostic_gemv(x, w, scale)
    reference = (x.float() @ (w.float() * scale).t()).to(dtype)
    assert out.dtype == dtype
    torch.testing.assert_close(out, reference, atol=0 if value is not None else 2e-3, rtol=0 if value is not None else 2e-2)

for dtype in [torch.float16, torch.bfloat16]:
    check(f'raw_gemv/{dtype}', raw_gemv, dtype)
    for m in [1, 4]:
        check(f'pair/m{m}/{dtype}', gemv.test_fp8_gemm_w8a16_pair, m, dtype)
        check(f'gated_norm/m{m}/{dtype}', norm.test_gated_rmsnorm_fp8_gemm, m, 16, 128, dtype)
        check(f'resadd_norm/m{m}/{dtype}', norm.test_resadd_rmsnorm_fp8_gemm, m, dtype)
        check(f'resadd_norm_pair/m{m}/{dtype}', norm.test_resadd_rmsnorm_fp8_gemm_pair, m, dtype)
        check(f'router/m{m}/{dtype}', router.test_router_gemv_topk_softmax, m, 256, 2048, 8, True, dtype)
        check(f'router_norm/m{m}/{dtype}', router.test_router_resadd_norm_gemv_topk_softmax, m, 256, 2048, 8, 'both', True, dtype)
    check(f'moe/m1/I256/{dtype}', moe.test_moe_shared_fused_decode, 1, 256, 0, dtype)
for value in [2.0**17, -(2.0**17), 2.0**-100]:
    check(f'gemv_bf16_range/{value}', raw_gemv, torch.bfloat16, value)
for m in [1, 4]:
    check(f'resadd_bf16_large/m{m}', norm.test_bf16_resadd_norm_keeps_large_residual, m)
check('moe/m4/I256/bf16/tile16', moe.test_moe_shared_fused_decode, 4, 256, 16, torch.bfloat16)
check('moe_reject_mixed_gate', moe.test_moe_shared_fused_decode_rejects_mixed_shared_gate_dtype)
print(f'{sum(r["passed"] for r in results)}/{len(results)} passed', flush=True)
raise SystemExit(0 if all(r['passed'] for r in results) else 1)
