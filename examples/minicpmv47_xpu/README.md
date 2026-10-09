# MiniCPM-V 4.7 BF16 / DSpark native 开发归档

2026-10-09：本分支以 kernels PR #622 的
`bee39d9299538ce6e82a947443790450aa570f3c` 为基线，保存当前 BF16 与 M16 修改。
保存归档前 `kernels-bf16` 的 10 个核心文件；提交钩子仅整理了一个测试文件的
import 顺序，其余文件逐字节一致。整理前的完整差异保留在
[original-source.patch](original-source.patch)，原始与发布源码哈希分别记录。

配套 vLLM 分支及完整性能、接受率、roofline 文档：
[xiangyuT/vllm:dev/minicpmv47-dspark-xpu-20261009](https://github.com/xiangyuT/vllm/tree/dev/minicpmv47-dspark-xpu-20261009/examples/minicpmv47_xpu)。

## 根目录修改

- FP8 GEMV、paired GEMV、norm-prologue GEMV 增加 BF16 激活与输出支持，使用对应 dtype 的原生实例化。
- routed/shared MoE 支持 BF16 激活，保留 FP8 per-tensor 权重与 FP32 scale 合同。
- router 与 residual-add/norm router 支持 BF16，small-M 上限从 8 扩至 16。
- Python dtype 门禁和现有 kernel 测试扩展覆盖上述行为。

M16 native 支持与 vLLM MoE compile-range 上限是不同设置；历史运行通过显式覆盖
选择 M16，配套 vLLM 默认融合阈值仍为 8。未修改包版本或默认启动配置。

## 本目录实验源文件

`functional-20261008/` 和 `performance-20261008/` 保存局部 BF16 addon 的构建、
绑定及检查；`dspark-gpu01-20261008/` 保存 head 256、paged non-causal attention
的专用编译配置和独立 op namespace 链接脚本；`dspark-iteration-20261009/`
保存 M16 router addon 构建；`dspark-optimization-20261009/` 保存 K=256 FP8
Markov GEMV 的隔离实验与 microbenchmark。这些实验没有全部集成进标准 wheel。

SYCL-TLA 固定 revision：`87f6850680a580654b9ea2c80dbc01aeb36ad231`。
历史编译使用 oneAPI 2026.1.1、Torch 2.14.0+xpu；标准构建参考仓库根目录指南。
实验脚本保留原始 `/workspace/preparation/kernels-bf16`、
`/workspace/build/kernels` 路径，需要原容器布局或路径调整。

DSpark checkpoint 的草稿 `block_size=15`，最新性能与模板诊断显式使用 k=3。
attention 编译配置的 page size=64 是另一个参数；配套文档记录 target hybrid
缓存的额外调整。详见 [dspark-block-size.json](dspark-block-size.json)。

历史 BF16 原生编译、35 项独立功能检查及专用 attention 12 项检查结果随相关
摘要保留。本次归档执行静态检查与 CPU dtype 合同测试，没有使用 GPU、重建完整
wheel 或重测性能。历史文档的状态表述属于原测量日期。

源码来源与 SHA-256 见 [archive-manifest.json](archive-manifest.json)，
本次检查见 [archive-validation.json](archive-validation.json)。模型、依赖树、
`.so`、对象文件和大型 trace 保留在原主机；性能与 roofline 的完整轻量归档位于配套 vLLM 分支。
