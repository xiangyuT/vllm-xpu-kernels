import json
import pathlib
import shlex
import subprocess

root = pathlib.Path('/workspace/preparation/kernels-bf16')
build = pathlib.Path('/workspace/build/kernels')
out = pathlib.Path('/workspace/functional-addon')
out.mkdir(exist_ok=True)
bindings = (root / 'csrc/xpu/torch_bindings.cpp').read_text()
selected = []
for begin, end in [(bindings.index('  xpu_ops.def(\n      "fp8_gemm_w8a16_pair'), bindings.index('\n#endif', bindings.index('  xpu_ops.def(\n      "fp8_gemm_w8a16_pair'))),
                   (bindings.index('  xpu_ops.def(\n      "moe_shared_fused_decode_interface'), bindings.index('\n#endif', bindings.index('  xpu_ops.def(\n      "moe_shared_fused_decode_interface')))]:
    selected.append(bindings[begin:end])
router = (root / 'csrc/moe/torch_bindings.cpp').read_text()
router = router[router.index('  m.def(\n      "router_gemv_topk_softmax'):router.index('  // Apply topk sigmoid')]
code = '''#include <torch/library.h>
#include "xpu/gemv/fp8_gemv_interface.h"
#include "xpu/grouped_gemm/moe_shared_fused_interface.h"
#include "moe/moe_ops.h"
TORCH_LIBRARY_FRAGMENT(_xpu_C, xpu_ops) {
  xpu_ops.def("bf16_diagnostic_gemv(Tensor x, Tensor w, Tensor scale) -> Tensor");
  xpu_ops.impl("bf16_diagnostic_gemv", torch::kXPU, &vllm::fp8_gemv::fp8_gemv_w8a16);
''' + '\n'.join(selected) + '\n}\nTORCH_LIBRARY_FRAGMENT(_moe_C, m) {\n' + router + '\n}\n'
src = out / 'bindings.cpp'
src.write_text(code)
rows = json.loads((build / 'compile_commands.json').read_text())
template = shlex.split(next(r['command'] for r in rows if 'moe_router_decode.cpp' in r['file']))
objects = []
for source in [src, root / 'csrc/xpu/grouped_gemm/moe_shared_fused_interface.cpp']:
    args = template.copy()
    target = out / (source.stem + '.o')
    args[args.index('-o') + 1] = str(target)
    args[-1] = str(source)
    args.insert(1, '-I' + str(root))
    print('COMPILE', source, flush=True)
    if not target.exists():
        subprocess.run(args, cwd=build, check=True)
    objects.append(str(target))
objects += [str(build / 'csrc/xpu/gemv/xe_2/CMakeFiles/gemv_kernels_xe_2.dir/fp8_gemv_xe2.cpp.o'),
            str(build / 'csrc/xpu/gemv/xe_2/CMakeFiles/gemv_kernels_xe_2.dir/norm_fp8_gemv_xe2.cpp.o'),
            str(build / 'csrc/xpu/grouped_gemm/xe_2/CMakeFiles/grouped_gemm_xe_2.dir/moe_shared_fused_decode_xe2.cpp.o'),
            str(build / 'CMakeFiles/_moe_C.dir/csrc/moe/moe_router_decode.cpp.o')]
torchlib = '/opt/venv/lib/python3.12/site-packages/torch/lib'
args = [template[0], '-shared', '-fsycl', '-fsycl-max-parallel-link-jobs=16',
        '-flink-huge-device-code', '-Xspirv-translator',
        '-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate',
        '-o', str(out / 'bf16-diagnostic.so'), *objects,
        '-L' + torchlib, '-Wl,-rpath,' + torchlib, '-ltorch', '-ltorch_cpu', '-ltorch_xpu', '-lc10', '-lc10_xpu', '-lze_loader']
print('LINK', flush=True)
subprocess.run(args, check=True)
print('DONE', flush=True)
