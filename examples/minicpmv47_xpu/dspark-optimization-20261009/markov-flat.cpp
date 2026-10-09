#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <ATen/DeviceGuard.h>
#include <torch/all.h>
#include "utils.h"

namespace vllm::fp8_markov_flat {

inline float decode(uint32_t byte) {
  const uint32_t sign = (byte & 128u) << 24;
  const uint32_t exp = (byte >> 3) & 15u;
  const uint32_t mant = byte & 7u;
  if (exp == 0) {
    float value = float(mant) * (1.0f / 512.0f);
    return (sign != 0) ? -value : value;
  }
  const uint32_t bits = (exp == 15 && mant == 7)
                            ? (sign | 0x7fc00000u)
                            : (sign | ((exp + 120u) << 23) | (mant << 20));
  return sycl::bit_cast<float>(bits);
}

template <typename Scalar, int Rows, int Threads>
struct FlatKernel {
  const Scalar* a;
  const uint32_t* w;
  const float* scale;
  Scalar* output;
  int n;
  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    auto sg = item.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int row0 = item.get_group(0) * (Threads / 16) * Rows +
                     sg.get_group_linear_id() * Rows;
    float av[16];
#pragma unroll
    for (int b = 0; b < 4; ++b) {
#pragma unroll
      for (int j = 0; j < 4; ++j) av[b*4+j] = float(a[(b*16+lane)*4+j]);
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
      const int row = row0 + r;
      float sum = 0;
      if (row < n) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          uint32_t word = w[row*64+b*16+lane];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            float value = decode((word >> (j*8)) & 255u);
            sum = sycl::fma(av[b*4+j],value,sum);
          }
        }
      }
      sum = sycl::reduce_over_group(sg,sum,sycl::plus<float>());
      if (lane == 0 && row < n) output[row] = Scalar(sum * scale[0]);
    }
  }
};

template <typename Scalar, int Rows, int Threads>
void launch(const torch::Tensor& a,const torch::Tensor& w,const torch::Tensor& s,
            torch::Tensor& out) {
  const int n = w.size(0);
  constexpr int tile = Rows * (Threads / 16);
  const int groups = (n + tile - 1) / tile;
  auto& q = vllm::xpu::vllmGetQueue();
  sycl::ext::oneapi::experimental::nd_launch(q,
    sycl::nd_range<1>(size_t(groups)*Threads,Threads),
    FlatKernel<Scalar,Rows,Threads>{reinterpret_cast<const Scalar*>(a.data_ptr()),
      reinterpret_cast<const uint32_t*>(w.data_ptr()),s.data_ptr<float>(),
      reinterpret_cast<Scalar*>(out.data_ptr()),n});
}

template <typename Scalar>
void choose(const torch::Tensor& a,const torch::Tensor& w,const torch::Tensor& s,
            torch::Tensor& out,int64_t policy) {
  switch (policy) {
    case 0: return launch<Scalar,4,256>(a,w,s,out);
    case 1: return launch<Scalar,8,256>(a,w,s,out);
    case 2: return launch<Scalar,16,256>(a,w,s,out);
    case 3: return launch<Scalar,4,128>(a,w,s,out);
    case 4: return launch<Scalar,8,128>(a,w,s,out);
    case 5: return launch<Scalar,16,128>(a,w,s,out);
    default: TORCH_CHECK(false,"invalid flat GEMV policy");
  }
}

torch::Tensor gemv(const torch::Tensor& a,const torch::Tensor& w,
                   const torch::Tensor& scale,int64_t policy) {
  TORCH_CHECK(a.is_xpu() && w.device()==a.device() && scale.device()==a.device(),
              "Markov GEMV requires same XPU device");
  TORCH_CHECK((a.scalar_type()==at::kBFloat16 || a.scalar_type()==at::kHalf) &&
              a.sizes()==at::IntArrayRef({1,256}) && a.is_contiguous(),
              "Markov GEMV requires contiguous FP16/BF16 [1,256]");
  TORCH_CHECK(w.dim()==2 && w.size(1)==256 && w.is_contiguous() &&
              w.scalar_type()==at::ScalarType::Float8_e4m3fn,
              "Markov GEMV requires contiguous FP8 [N,256]");
  TORCH_CHECK(scale.numel()==1 && scale.scalar_type()==at::kFloat,
              "Markov GEMV requires one FP32 scale");
  const at::DeviceGuard guard(a.device());
  auto out=at::empty({1,w.size(0)},a.options());
  if (a.scalar_type()==at::kHalf) choose<sycl::half>(a,w,scale,out,policy);
  else choose<sycl::ext::oneapi::bfloat16>(a,w,scale,out,policy);
  return out;
}
}
