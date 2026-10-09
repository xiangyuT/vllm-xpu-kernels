import hashlib
import json
from pathlib import Path
import shlex
import subprocess

source=Path('/workspace/preparation/kernels-bf16/csrc/moe/moe_router_decode.cpp')
out=Path('/workspace/m16-router-20261009')
out.mkdir(exist_ok=True)
code=source.read_text().replace('namespace moe {','namespace moe_m16 {')
code=code.replace('using namespace vllm::moe;','using namespace vllm::moe_m16;')
code=code.replace('router_resadd_norm_gemv_topk_softmax','minicpm_m16_router_resadd_norm_gemv_topk_softmax')
code=code.replace('router_gemv_topk_softmax','minicpm_m16_router_gemv_topk_softmax')
code='#include <torch/library.h>\n'+code+'''
TORCH_LIBRARY_IMPL(_moe_C, XPU, m) {
  m.impl("router_gemv_topk_softmax", &minicpm_m16_router_gemv_topk_softmax);
  m.impl("router_resadd_norm_gemv_topk_softmax", &minicpm_m16_router_resadd_norm_gemv_topk_softmax);
}
'''
generated=out/'router.cpp'
generated.write_text(code)
build=Path('/workspace/build/kernels')
rows=json.loads((build/'compile_commands.json').read_text())
args=shlex.split(next(row['command'] for row in rows if 'moe_router_decode.cpp' in row['file']))
compiler=args[0]
args[args.index('-o')+1]=str(out/'router.o')
args[-1]=str(generated)
args.insert(1,'-I'+str(source.parent))
print('COMPILE M16 router',flush=True)
subprocess.run(args,check=True,cwd=build)
torchlib='/opt/venv/lib/python3.12/site-packages/torch/lib'
target=out/'m16-router.so'
link=[compiler,'-shared','-fsycl','-flink-huge-device-code',
 '-Xspirv-translator','-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io',
 '-o',str(target),str(out/'router.o'),'-L'+torchlib,'-Wl,-rpath,'+torchlib,
 '-ltorch','-ltorch_cpu','-ltorch_xpu','-lc10','-lc10_xpu','-lze_loader']
print('LINK',flush=True)
subprocess.run(link,check=True)
(out/'manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'generated_sha256':hashlib.sha256(generated.read_bytes()).hexdigest(),
 'library_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
 'namespace':'vllm::moe_m16','entrypoints':'minicpm_m16_*','gpu_execution':False},indent=2)+'\n')
print('COMPLETE',flush=True)
