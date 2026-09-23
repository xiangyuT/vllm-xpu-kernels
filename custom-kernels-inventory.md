# Custom kernel addition on the Qwen3.8 FP8 decode branch

`_xpu_C.fp8_gemm_block_decode(A, A_quant, B, A_scale, B_scale)` returns an
FP16 matrix product. `A` is the original FP16 activation; `A_quant` is its
dynamic FP8 quantization. `B` is a transposed E4M3 weight with 128×128 block
scales. At runtime the operator uses the existing W8A16 GEMM for one input
token and the existing W8A8 GEMM for other token counts. Activation
quantization still runs for both paths.

The paired vLLM provider enables this operator only for aligned, non-BMM
128×128 block-scaled layers when `VLLM_XPU_FP8_BLOCK_HYBRID=1`. It registers
a fake implementation for `torch.compile` and includes the flag in the XPU
compilation cache hash. No dependency or package version changes are needed.
