#!/usr/bin/env bash
set -euo pipefail
set +u
source /opt/intel/oneapi/setvars.sh >/dev/null
set -u
cmake_bin=/opt/venv/bin/cmake
build_dir=/workspace/dspark-attention-build
"$cmake_bin" -S /workspace/preparation/kernels-bf16 -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx \
    -DVLLM_PYTHON_EXECUTABLE=/workspace/tools/.venv/bin/python \
    -DVLLM_CUTLASS_SRC_DIR=/workspace/preparation/bf16-deps/sycl-tla \
    -DFETCHCONTENT_BASE_DIR=/workspace/dspark-attention-deps \
    -DVLLM_XPU_ENABLE_XE2=ON -DVLLM_XPU_ENABLE_XE_DEFAULT=OFF \
    -DVLLM_XPU_ENABLE_XE3P=OFF -DVLLM_XPU_ENABLE_ONEDNN=OFF \
    -DBASIC_KERNELS_ENABLED=OFF -DFA2_KERNELS_ENABLED=ON \
    -DMOE_KERNELS_ENABLED=OFF -DGDN_KERNELS_ENABLED=OFF \
    -DMQA_LOGITS_KERNELS_ENABLED=OFF -DMHC_KERNELS_ENABLED=OFF \
    -DXPU_SPECIFIC_KERNELS_ENABLED=OFF -DXPUMEM_ALLOCATOR_ENABLED=OFF \
    -DVLLM_CHUNK_PREFILL_CONFIG=/workspace/preparation/dspark-gpu01-20261008/chunk-prefill-dspark.conf \
    -DVLLM_PAGED_DECODE_CONFIG=/workspace/preparation/dspark-gpu01-20261008/paged-decode-dspark.conf
"$cmake_bin" --build "$build_dir" --target attn_kernels_xe_2 -j 4
/workspace/tools/.venv/bin/python /workspace/preparation/dspark-gpu01-20261008/link-dspark-attention.py
