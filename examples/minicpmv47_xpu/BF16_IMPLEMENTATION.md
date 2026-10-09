# MiniCPM V4.7 XPU BF16 优化实现与 CPU 检查

> 2026-10-08 性能更新：BF16 原生 overlay 已在 MiniCPM 实际 decode 中执行，32K/512 TPOT 中位数 17.724 ms/token；源码/镜像/overlay 边界及未命中的 GDN pass 见 [性能采集与历史对照](PERFORMANCE_COMPARISON_20261008.md)。此前“尚未证明端到端收益”描述保留为无 GPU 阶段的历史状态，当前仍缺完整候选 wheel、单项消融和系统准确率评测。

> 2026-10-08 后续更新：已获准使用物理卡 0/3 完成首轮功能检查，独立原生内核 35/35 项通过；main 与模型 PR 的集成源码也已通过旧权重 TP=2 文本检查。详情见 [首轮功能验证](FUNCTIONAL_VALIDATION_20261008.md)。下文保留此前无 GPU 阶段的证据与边界；模型编译融合命中和性能仍未验证。

2026-10-08。已在原性能 PR 的两个独立本地开发分支上补充 **BF16 激活、FP8 权重**的代码路径，覆盖 MoE 与 shared expert 融合、router、FP8 GEMV 和 norm 与投影融合。BF16 输入、输出和中间存储保留 BF16，GEMV 选择 BF16 DPAS，累加仍为 FP32。当前证据是源码、CPU 测试和无 GPU 的原生编译，尚未证明 GPU 数值正确性或 MiniCPM 端到端收益。

## 分支和精确改动

| 仓库工作树 | 本地分支 | 基础提交 |
| --- | --- | --- |
| [vllm-bf16](vllm-bf16) | `dev/minicpmv47-bf16-fusions-20261008` | `8f3e461beebc37d665a2cc2ba805adea1617dc34`，PR #60539 head |
| [kernels-bf16](kernels-bf16) | `dev/minicpmv47-bf16-kernels-20261008` | `bee39d9299538ce6e82a947443790450aa570f3c`，PR #622 head |

两组改动均未提交。完整差异分别为 [vllm-bf16-final.patch](bf16-evidence/vllm-bf16-final.patch) 和 [kernels-bf16-final.patch](bf16-evidence/kernels-bf16-final.patch)，每个改动文件的哈希、基础提交及 patch 哈希保存在 [implementation-manifest.json](bf16-evidence/implementation-manifest.json)。vLLM 修改 5 个文件，kernels 修改 10 个文件；没有修改版本、创建新提交或远程写入。

## BF16 路径

| 部分 | 实现 | 主要文件 |
| --- | --- | --- |
| vLLM 配置与图匹配 | MoE 融合和 decode compile range 接受 FP16/BF16；输入 norm 融合要求 x、residual、norm weight 与 hidden dtype 一致；router 与 shared gate 也须匹配模型激活 dtype | `vllm/config/vllm.py`、`vllm/compilation/passes/fusion/xpu_fusion.py`、`vllm/_xpu_ops.py` |
| MoE 与 shared expert | gather、两次 GEMM、SiLU/shared gate 和 finalize 按 activation 类型实例化，BF16 使用 `cutlass::bfloat16_t` 与对应 MMA；workspace 仍按两字节元素布局 | `csrc/xpu/grouped_gemm/xe_2/moe_shared_fused_decode_xe2.cpp` |
| router 与输入 norm | 输入、权重、SLM norm 存储、输出与 logits 舍入按 dtype 分派；top-k 排序编码和解码也使用对应的 BF16 位表示，避免把 BF16 logits 经 FP16 截断 | `csrc/moe/moe_router_decode.cpp` |
| FP8 GEMV 与双投影 | 2D load 继续读取两字节激活；选择 BF16 DPAS 和 BF16 output；FP8 E4M3 解码处理 BF16 指数偏置、次正规数、符号零和 NaN | `csrc/xpu/gemv/xe_2/fp8_gemv.hpp`、`fp8_gemv_xe2.cpp` |
| norm 与 GEMV | gated/residual-add norm 的寄存器 prologue、generic prologue 和多行独立 SYCL norm 都按 FP16/BF16 实例化；BF16 norm 输出直接送 BF16 GEMV | `csrc/xpu/gemv/xe_2/norm_fp8_gemv_xe2.cpp` |
| Python kernel 入口 | MoE 与 router 的公开支持判断接受 BF16；dtype 不匹配由调用前门禁或原生入口拒绝 | `vllm_xpu_kernels/moe_shared_fused_interface.py` |

原有形状、量化、设备及并行门禁保留。vLLM MoE 融合仍要求 hidden 2048、256 experts、top-8、per-tensor FP8 和匹配的 shared expert 等条件。自动 GEMV dispatch 仍限于 BMG、M=1 及原有 layout/alignment 条件；M>1 保留 oneDNN GEMM，norm 可以使用独立 BF16 SYCL kernel。QKV norm 与 RoPE、in-place all-reduce 和 GDN allocation 原先没有同样的 FP16 限制，本轮未改写这些路径。

BF16 的 FP8 解码比原来的 FP16 位移公式多了指数与次正规数处理。相同融合结构不等于相同性能，BF16 的实际收益必须后续测量。

## 已完成检查

- **59 个 vLLM CPU/metadata 测试通过**：FP16/BF16 MoE 图改写、norm 融合、混合 dtype 回退、router/shared gate dtype 门禁、online FP8 门禁和 compile range 等。测试使用显式 CPU platform、无 accelerator 的 cleanup fixture；图改写使用同 schema 的占位 op，门禁测试模拟 kernel 可用性。这些不是 GPU op 数值测试。[结果](bf16-evidence/vllm-cpu-tests-attempt2.log)
- **4 个 kernels Python dtype 合同测试通过**：FP16/BF16 接受、FP32/FP64 拒绝；仅模拟 native op 存在，不执行它。[结果](bf16-evidence/kernels-cpu-tests.log)
- **CPU 数值编码检查通过**：全部 256 个 E4M3→BF16 码值，包括符号零和 NaN；254 个有限 E4M3→FP16 码值；1024 个打包字，以及 `2^17`、`-2^17`、`2^-100` 的 BF16 范围检查。使用实际生产 header，与独立浮点数参考公式对照。[结果](bf16-evidence/cpu-codec-check.log)
- **原生编译和链接通过**：oneAPI 2026.1.1、Torch 2.14.0+xpu、固定 SYCL-TLA `87f6850680a580654b9ea2c80dbc01aeb36ad231`，构建 `gemv_kernels_xe_2`、`grouped_gemm_xe_2` 和 `_moe_C`，包含 FP16/BF16 实例化。最后一次对格式整理后的源码重建也通过。[结果](bf16-evidence/native-build-final.log)
- **源码检查通过**：10 个改动 Python 文件的 AST 解析、Ruff 检查、vLLM Ruff format、clang-format 20.1.3 检查，以及两仓库的 `git diff --check`。开发工作树已安装 pre-commit hooks。[格式结果](bf16-evidence/final-lint-attempt2.log)

初次 CPU pytest 缺少被 `confcutdir` 排除的 fixture，并无法在无设备环境自动推断平台，属于测试启动配置问题；改用显式 CPU platform 与对应 fixture 后通过，原失败日志保留。原生编译中的 SYCL subgroup load deprecation 来自沿用的上游代码，没有将其解释为镜像或候选失败。

## GPU 检查与部署边界

按用户要求，本轮未创建 XPU 张量、调用 GPU kernel、启动推理服务或做 benchmark。编译容器 `codex-minicpm47-bf16-build-20261008` 的完整 ID 为 `fb05fa28e23be50b5f71ed365bf75890a9b4240b158785cf7cf3d453abce0b2a`，使用先前准备的 image `sha256:c1ae7c6509f6cda9341c15fd947da0000faa4a5d437af651d7f5018689818566`；源码只读挂载，无 GPU device request、无 `/dev/dri` 或模型挂载。[容器身份](bf16-evidence/build-container-identity.json)

原生检查使用空 AOT device lists，并关闭 oneDNN 和无关 kernel families，以编译改动的目标。它没有产出可交付的完整 `_xpu_C`、wheel 或包含候选代码的新镜像；原装开发容器中的 vLLM 和 kernels 仍保持此前版本。

已扩展原有 GPU 数值 suites 的 BF16 参数，覆盖 M=1/2/3/4/8、多种专家 tile、int32/int64 routing ids、strided norm 输入、in-place router norm、不同 streams、混合 dtype 拒绝，以及超过 FP16 范围的 GEMV/router/residual 数值。这些 GPU 测试本轮未运行。后续仍需在实际目标设备执行这些 suites，再确认 MiniCPM 4.7 的真实融合命中、文本/多模态正确性和对齐配置的性能比较。

本轮专注补充原性能 PR 的 BF16 路径，尚未把 MiniCPM 模型 PR #58674 与 main 完成整合。原 v0.30 工作树、阅读用 main/PR 快照、权重、用户媒体和既有容器保留。源码存档目标继续采用用户指定的 `xiangyuT/vllm`，尚未推送。
