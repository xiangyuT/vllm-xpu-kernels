"""Build flat GEMV plus an independent original oneDNN entry for comparison."""

import hashlib
import json
import shlex
import subprocess
from pathlib import Path

PREP = Path("/workspace/preparation")
ROOT = PREP / "dspark-optimization-20261009"
BUILD = Path("/workspace/build/kernels")
OUT = Path("/workspace/markov-flat-addon")
OUT.mkdir(exist_ok=True)
bindings = OUT / "bindings.cpp"
bindings.write_text("""#include <torch/all.h>
#include "xpu/ops.h"
namespace vllm::fp8_markov_flat {
torch::Tensor gemv(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,int64_t);
}
torch::Tensor original_onednn(const torch::Tensor& a,const torch::Tensor& w,
                             const torch::Tensor& scale) {
  return ::fp8_gemm_w8a16(a,w.t(),scale,std::nullopt);
}
TORCH_LIBRARY(_minicpm_markov2_C,m) {
  m.def("flat(Tensor a, Tensor w, Tensor scale, int policy) -> Tensor");
  m.def("onednn(Tensor a, Tensor w, Tensor scale) -> Tensor");
}
TORCH_LIBRARY_IMPL(_minicpm_markov2_C,XPU,m) {
  m.impl("flat",&vllm::fp8_markov_flat::gemv);
  m.impl("onednn",&original_onednn);
}
""")
commands = json.loads((BUILD / "compile_commands.json").read_text())
template = shlex.split(
    next(r["command"] for r in commands if r["file"].endswith("fp8_gemv_xe2.cpp"))
)
objects = []
for path in [ROOT / "markov-flat.cpp", bindings]:
    obj = OUT / (path.stem + ".o")
    args = template.copy()
    args[args.index("-o") + 1] = str(obj)
    args[-1] = str(path)
    for flag in ["-MT", "-MF"]:
        if flag in args:
            i = args.index(flag)
            del args[i : i + 2]
    if "-MD" in args:
        args.remove("-MD")
    print("COMPILE", path, flush=True)
    subprocess.run(args, check=True, cwd=BUILD)
    objects.append(str(obj))
torchlib = "/opt/venv/lib/python3.12/site-packages/torch/lib"
kernellib = "/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels"
library = OUT / "markov-flat.so"
args = [
    template[0],
    "-shared",
    "-fsycl",
    "-fsycl-max-parallel-link-jobs=16",
    "-flink-huge-device-code",
    "-Xspirv-translator",
    "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate",
    "-o",
    str(library),
    *objects,
    str(Path(kernellib) / "_xpu_C.abi3.so"),
    "-L" + torchlib,
    "-Wl,-rpath," + torchlib + ":" + kernellib,
    "-ltorch",
    "-ltorch_cpu",
    "-ltorch_xpu",
    "-lc10",
    "-lc10_xpu",
    "-lze_loader",
]
print("LINK", flush=True)
subprocess.run(args, check=True)
(OUT / "manifest.json").write_text(
    json.dumps(
        {
            "namespace": "vllm::fp8_markov_flat",
            "source_sha256": hashlib.sha256(
                (ROOT / "markov-flat.cpp").read_bytes()
            ).hexdigest(),
            "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "policies": {
                "0": "rows4 WG256",
                "1": "rows8 WG256",
                "2": "rows16 WG256",
                "3": "rows4 WG128",
                "4": "rows8 WG128",
                "5": "rows16 WG128",
                "onednn": "original stock C++ FP8 matmul, bypass generic GEMV",
            },
            "scope": "FP16/BF16 [1,256] and FP8 [N,256], no package/version changes",
        },
        indent=2,
    )
    + "\n"
)
print("DONE", flush=True)
