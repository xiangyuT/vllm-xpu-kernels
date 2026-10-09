import json
from pathlib import Path
import shlex
import subprocess

source = Path('/workspace/preparation/kernels-bf16')
build = Path('/workspace/build/kernels')
out = Path('/workspace/performance-addon')
out.mkdir(exist_ok=True)
old = Path('/workspace/functional-addon')
bindings = (source / 'csrc/xpu/torch_bindings.cpp').read_text()
begin = bindings.index('  xpu_ops.def(\n      "qkv_split_norm_rope')
end = bindings.index('\n}', begin)
code = (old / 'bindings.cpp').read_text()
code = '#include "xpu/ops.h"\n' + code
code += '\nTORCH_LIBRARY_FRAGMENT(_xpu_C, xpu_ops) {\n' + bindings[begin:end] + '\n}\n'
code += '''
torch::Tensor diagnostic_fp8_dispatch(
    const torch::Tensor& a, const torch::Tensor& b,
    const std::optional<torch::Tensor>& scale,
    const std::optional<torch::Tensor>& bias) {
  if (auto out = vllm::fp8_gemv::try_fp8_gemv_w8a16(a, b, scale, bias))
    return *out;
  // Call the original oneDNN C++ implementation directly, avoiding a
  // dispatcher fallback to this replacement registration.
  return ::fp8_gemm_w8a16(a, b, scale, bias);
}
TORCH_LIBRARY_IMPL(_xpu_C, XPU, m) {
  m.impl("fp8_gemm_w8a16", &diagnostic_fp8_dispatch);
}
'''
src = out / 'bindings.cpp'
src.write_text(code)
rows = json.loads((build / 'compile_commands.json').read_text())
template = shlex.split(next(row['command'] for row in rows if 'moe_router_decode.cpp' in row['file']))
objects = []
for path in [src, source / 'csrc/xpu/sycl/qkv_split_norm_rope.cpp']:
    target = out / (path.stem + '.o')
    args = template.copy()
    args[args.index('-o') + 1] = str(target)
    args[-1] = str(path)
    args.insert(1, '-I' + str(source))
    print('COMPILE', path, flush=True)
    subprocess.run(args, check=True, cwd=build)
    objects.append(str(target))
objects += [str(old / 'moe_shared_fused_interface.o'),
            str(build / 'csrc/xpu/gemv/xe_2/CMakeFiles/gemv_kernels_xe_2.dir/fp8_gemv_xe2.cpp.o'),
            str(build / 'csrc/xpu/gemv/xe_2/CMakeFiles/gemv_kernels_xe_2.dir/norm_fp8_gemv_xe2.cpp.o'),
            str(build / 'csrc/xpu/grouped_gemm/xe_2/CMakeFiles/grouped_gemm_xe_2.dir/moe_shared_fused_decode_xe2.cpp.o'),
            str(build / 'CMakeFiles/_moe_C.dir/csrc/moe/moe_router_decode.cpp.o')]
torchlib = '/opt/venv/lib/python3.12/site-packages/torch/lib'
kernellib = '/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels'
args = [template[0], '-shared', '-fsycl', '-fsycl-max-parallel-link-jobs=16',
        '-flink-huge-device-code', '-Xspirv-translator',
        '-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate',
        '-o', str(out / 'bf16-performance.so'), *objects,
        str(Path(kernellib) / '_xpu_C.abi3.so'),
        '-L' + torchlib, '-Wl,-rpath,' + torchlib + ':' + kernellib,
        '-ltorch', '-ltorch_cpu', '-ltorch_xpu', '-lc10', '-lc10_xpu', '-lze_loader']
print('LINK', flush=True)
subprocess.run(args, check=True)
print('DONE', flush=True)
