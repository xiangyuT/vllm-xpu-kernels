"""Build isolated K=256 GEMV policies with the established oneAPI commands."""

import hashlib
import json
import shlex
import subprocess
from pathlib import Path

SOURCE = Path("/workspace/preparation/kernels-bf16/csrc/xpu/gemv")
BUILD = Path("/workspace/build/kernels")
OUT = Path("/workspace/markov-addon")
(OUT / "gemv/xe_2").mkdir(parents=True, exist_ok=True)
namespace = "vllm::fp8_markov_tuning"
for rel in ["fp8_gemv_interface.h", "xe_2/fp8_gemv.hpp", "xe_2/fp8_gemv_xe2.cpp"]:
    text = (SOURCE / rel).read_text().replace("vllm::fp8_gemv", namespace)
    if rel.endswith(".cpp"):
        text = text.replace(
            "#include <optional>", "#include <optional>\n#include <atomic>"
        )
        text = text.replace(
            f"namespace {namespace} {{",
            f"""namespace {namespace} {{
static std::atomic<int64_t> markov_config{{0}};
void set_markov_config(int64_t value) {{
  TORCH_CHECK(value >= 0 && value <= 4, "invalid Markov policy");
  markov_config.store(value);
}}
""",
            1,
        )
        needle = "  if (n_total <= 512)\n"
        assert text.count(needle) == 1
        text = text.replace(
            needle,
            """  if (M == 1 && K == 256 && n_total >= 65536) {
    switch (markov_config.load()) {
      case 1: return run_cfg<1, 4, NSEG>(q, a, a_pitch_bytes, M, K, s0, s1);
      case 2: return run_cfg<1, 8, NSEG>(q, a, a_pitch_bytes, M, K, s0, s1);
      case 3: return run_cfg<1, 16, NSEG>(q, a, a_pitch_bytes, M, K, s0, s1);
      case 4: return run_cfg<2, 4, NSEG>(q, a, a_pitch_bytes, M, K, s0, s1);
      default: break;
    }
  }
"""
            + needle,
            1,
        )
    (OUT / "gemv" / rel).write_text(text)
binding = OUT / "bindings.cpp"
binding.write_text(f"""#include <torch/all.h>
#include "gemv/fp8_gemv_interface.h"
namespace {namespace} {{ void set_markov_config(int64_t); }}
TORCH_LIBRARY(_minicpm_markov_C, m) {{
  m.def("gemv(Tensor x, Tensor w, Tensor scale) -> Tensor");
  m.def("set_config(int value) -> ()", &{namespace}::set_markov_config);
}}
TORCH_LIBRARY_IMPL(_minicpm_markov_C, XPU, m) {{
  m.impl("gemv", &{namespace}::fp8_gemv_w8a16);
}}
""")
commands = json.loads((BUILD / "compile_commands.json").read_text())
template = shlex.split(
    next(r["command"] for r in commands if r["file"].endswith("fp8_gemv_xe2.cpp"))
)
objects = []
for path in [OUT / "gemv/xe_2/fp8_gemv_xe2.cpp", binding]:
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
    args.insert(1, "-I" + str(OUT))
    print("COMPILE", path, flush=True)
    subprocess.run(args, check=True, cwd=BUILD)
    objects.append(str(obj))
torchlib = "/opt/venv/lib/python3.12/site-packages/torch/lib"
library = OUT / "markov-tuning.so"
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
    "-L" + torchlib,
    "-Wl,-rpath," + torchlib,
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
            "namespace": namespace,
            "policies": {
                "0": "legacy KS4/RG1",
                "1": "KS1/RG4",
                "2": "KS1/RG8",
                "3": "KS1/RG16",
                "4": "KS2/RG4",
            },
            "sources": {
                str(SOURCE / r): hashlib.sha256((SOURCE / r).read_bytes()).hexdigest()
                for r in [
                    "fp8_gemv_interface.h",
                    "xe_2/fp8_gemv.hpp",
                    "xe_2/fp8_gemv_xe2.cpp",
                ]
            },
            "generated": {
                str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    binding,
                    OUT / "gemv/xe_2/fp8_gemv_xe2.cpp",
                    OUT / "gemv/xe_2/fp8_gemv.hpp",
                    OUT / "gemv/fp8_gemv_interface.h",
                ]
            },
            "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "scope": "M1/K256/N>=65536 only; other policies unchanged; isolated op, no wheel/version changes",
        },
        indent=2,
    )
    + "\n"
)
print("DONE", flush=True)
