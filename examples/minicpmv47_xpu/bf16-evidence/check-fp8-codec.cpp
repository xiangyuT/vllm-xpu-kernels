#include "csrc/xpu/gemv/xe_2/fp8_gemv.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

using BFloat16 = sycl::ext::oneapi::bfloat16;

float reference(uint32_t byte) {
  const uint32_t magnitude = byte & 0x7f;
  const int exponent = magnitude >> 3;
  const int mantissa = magnitude & 7;
  if (magnitude == 0x7f) return std::nanf("");
  const float value = exponent == 0 ? std::ldexp(float(mantissa), -9)
                                   : std::ldexp(1.f + mantissa / 8.f, exponent - 7);
  return std::copysign(value / 256.f, byte & 0x80 ? -1.f : 1.f);
}

void check_bf16(uint16_t bits, uint32_t byte) {
  uint32_t float_bits = uint32_t(bits) << 16;
  float actual;
  std::memcpy(&actual, &float_bits, sizeof(actual));
  const float expected = reference(byte);
  if (std::isnan(expected)) {
    if (!std::isnan(actual)) throw std::runtime_error("NaN lost");
  } else if (actual != expected || std::signbit(actual) != std::signbit(expected)) {
    throw std::runtime_error("BF16 decode differs from E4M3 reference");
  }
}

int main() {
  using namespace vllm::fp8_gemv;
  for (uint32_t byte = 0; byte < 256; ++byte) {
    check_bf16(e4m3_bf16(byte), byte);
    if ((byte & 0x7f) != 0x7f) {
      const uint16_t bits = e4m3_even<>(byte);
      const float decoded = float(sycl::bit_cast<sycl::half>(bits));
      if (decoded != reference(byte) || std::signbit(decoded) != std::signbit(reference(byte)))
        throw std::runtime_error("FP16 decode regressed");
    }
    for (uint32_t position = 0; position < 4; ++position) {
      const uint32_t word = (0x08070605u & ~(0xffu << (8 * position))) |
                            (byte << (8 * position));
      const uint32_t even = e4m3_even<BFloat16>(word);
      const uint32_t odd = e4m3_odd<BFloat16>(word);
      check_bf16(even & 0xffffu, word & 0xffu);
      check_bf16(even >> 16, (word >> 16) & 0xffu);
      check_bf16(odd & 0xffffu, (word >> 8) & 0xffu);
      check_bf16(odd >> 16, word >> 24);
    }
  }
  for (float value : {std::ldexp(1.f, 17), -std::ldexp(1.f, 17), std::ldexp(1.f, -100)}) {
    if (float(BFloat16(value)) != value)
      throw std::runtime_error("BF16 range lost");
  }
  std::cout << "CPU_CODEC_PASS: 256 BF16 codes, 254 finite FP16 codes, "
               "1024 packed words, 3 BF16 range values\n";
}
