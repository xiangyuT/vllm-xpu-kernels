#include "xpu/ops.h"
#include <torch/library.h>
#include "xpu/gemv/fp8_gemv_interface.h"
#include "xpu/grouped_gemm/moe_shared_fused_interface.h"
#include "moe/moe_ops.h"
TORCH_LIBRARY_FRAGMENT(_xpu_C, xpu_ops) {
  xpu_ops.def("bf16_diagnostic_gemv(Tensor x, Tensor w, Tensor scale) -> Tensor");
  xpu_ops.impl("bf16_diagnostic_gemv", torch::kXPU, &vllm::fp8_gemv::fp8_gemv_w8a16);
  xpu_ops.def(
      "fp8_gemm_w8a16_pair(Tensor A, Tensor B1, Tensor B1_scale, Tensor B2, "
      "Tensor B2_scale) -> (Tensor, Tensor)");
  xpu_ops.impl(
      "fp8_gemm_w8a16_pair", torch::kXPU, &vllm::fp8_gemv::fp8_gemm_w8a16_pair);
  // RMSNorm-prologue + fp8 linear fusions (one GEMV launch for decode rows,
  // unfused norm + fp8_gemm_w8a16 otherwise).
  xpu_ops.def(
      "gated_rmsnorm_fp8_gemm(Tensor x, Tensor z, Tensor norm_weight, "
      "float eps, Tensor B, Tensor B_scale) -> Tensor");
  xpu_ops.impl(
      "gated_rmsnorm_fp8_gemm",
      torch::kXPU,
      &vllm::fp8_gemv::gated_rmsnorm_fp8_gemm);
  xpu_ops.def(
      "resadd_rmsnorm_fp8_gemm(Tensor x, Tensor residual, Tensor "
      "norm_weight, float eps, Tensor B, Tensor B_scale) -> (Tensor, Tensor)");
  xpu_ops.impl(
      "resadd_rmsnorm_fp8_gemm",
      torch::kXPU,
      &vllm::fp8_gemv::resadd_rmsnorm_fp8_gemm);
  xpu_ops.def(
      "resadd_rmsnorm_fp8_gemm_pair(Tensor x, Tensor residual, Tensor "
      "norm_weight, float eps, Tensor B1, Tensor B1_scale, Tensor B2, Tensor "
      "B2_scale) -> (Tensor, Tensor, Tensor)");
  xpu_ops.impl(
      "resadd_rmsnorm_fp8_gemm_pair",
      torch::kXPU,
      &vllm::fp8_gemv::resadd_rmsnorm_fp8_gemm_pair);
  xpu_ops.def(
      "moe_shared_fused_decode_interface(Tensor(a!) output, Tensor x, "
      "Tensor topk_ids, Tensor topk_weights, Tensor w13, Tensor w13_scale, "
      "Tensor w2, Tensor w2_scale, Tensor shared_w13, Tensor "
      "shared_w13_scale, Tensor shared_w2, Tensor shared_w2_scale, Tensor "
      "shared_gate, Tensor(b!) ws, int tile_m=0) -> ()");
  xpu_ops.impl(
      "moe_shared_fused_decode_interface",
      torch::kXPU,
      &moe_shared_fused_decode_interface);
}
TORCH_LIBRARY_FRAGMENT(_moe_C, m) {
  m.def(
      "router_gemv_topk_softmax(Tensor x, Tensor router_weight, Tensor! "
      "logits, Tensor! topk_weights, Tensor! topk_ids, bool renormalize) -> "
      "()");
  m.impl("router_gemv_topk_softmax", torch::kXPU, &router_gemv_topk_softmax);
  // Same with a fused residual-add + Gemma (1 + w) RMSNorm prologue; writes
  // the normed input and the new residual (either may alias its input).
  m.def(
      "router_resadd_norm_gemv_topk_softmax(Tensor x, Tensor residual, "
      "Tensor norm_weight, float eps, Tensor router_weight, Tensor! logits, "
      "Tensor! topk_weights, Tensor! topk_ids, Tensor! normed_out, Tensor! "
      "residual_out, bool renormalize) -> ()");
  m.impl(
      "router_resadd_norm_gemv_topk_softmax",
      torch::kXPU,
      &router_resadd_norm_gemv_topk_softmax);

}

TORCH_LIBRARY_FRAGMENT(_xpu_C, xpu_ops) {
  xpu_ops.def(
      "qkv_split_norm_rope(Tensor qkv, Tensor positions, Tensor q_weight, "
      "Tensor k_weight, Tensor cos_sin_cache, Tensor! q_out, Tensor! k_out, "
      "Tensor!? gate_out, int num_q_heads, int num_kv_heads, int head_dim, "
      "int rotary_dim, float eps, float weight_offset, int[] mrope_section, "
      "bool mrope_interleaved) -> ()");
  xpu_ops.impl("qkv_split_norm_rope", torch::kXPU, &qkv_split_norm_rope);
}

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
