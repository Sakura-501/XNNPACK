# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for fp16 x86 dot kernel generators."""

# pylint: disable=invalid-name
# pylint: disable=missing-class-docstring

from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels
from ynnpack.kernels.dot.generator.x86 import x86
from ynnpack.kernels.dot.generator.x86 import x86_avx
from ynnpack.kernels.dot.generator.x86 import x86_avx512


class x86_fp16_fp16_fp32(x86):
  def __init__(self, arch, bits, tile_shape):
    super().__init__(arch, "fp16_fp16_fp32", "float", bits, tile_shape)
    self.a_type = "half"
    self.b_type = "half"

  def header(self):
    return super().header() + """

namespace {

using half = int16_t;

}  // namespace
"""

  def load_a_tile(self, i, k):
    a = f"{self._mm(self.bits // 2)}_set1_epi16(*{self.a_ptr(i, k)})"
    return f"__m{self.bits} a_{i}_{k} = {self._mm()}_cvtph_ps({a});\n"

  def load_b_tile(self, k, j):
    # Since we are converting f16 (16-bit) to f32 (32-bit), the input data
    # occupies exactly half the width of the output vector.
    b_ptr = self.b_ptr(k, j, f"__m{self.bits//2}i")
    b = f"{self._mm(self.bits//2)}_loadu_si{self.bits//2}({b_ptr})"
    return f"__m{self.bits} b_{k}_{j} = {self._mm()}_cvtph_ps({b});\n"

  def product(self, i, j, k):
    mm = self._mm()
    c_ij = f"c_{i}_{j}"
    return f"{c_ij} = {mm}_add_ps({c_ij}, {mm}_mul_ps(a_{i}_{k}, b_{k}_{j}));\n"


class x86_f16c_fp16_fp16_fp32(x86_fp16_fp16_fp32, x86_avx):
  def __init__(self):
    super().__init__(arch="f16c", bits=256, tile_shape=(1, 8, 1))
    self.flags += ["dot_flag::consistent_arithmetic"]


class x86_f16c_fma3_fp16_fp16_fp32(x86_fp16_fp16_fp32, x86_avx):

  def __init__(self):
    super().__init__(arch="fma3", bits=256, tile_shape=(1, 8, 1))
    self.flags += ["dot_flag::consistent_arithmetic"]

  def product(self, i, j, k):
    c_ij = f"c_{i}_{j}"
    return f"{c_ij} = {self._mm()}_fmadd_ps(a_{i}_{k}, b_{k}_{j}, {c_ij});\n"


class x86_avx512_fp16_fp16_fp32(x86_fp16_fp16_fp32, x86_avx512):

  def __init__(self):
    super().__init__(arch="avx512", bits=512, tile_shape=(1, 16, 1))
    self.flags += ["dot_flag::consistent_arithmetic"]

  def product(self, i, j, k):
    c_ij = f"c_{i}_{j}"
    return f"{c_ij} = {self._mm()}_fmadd_ps(a_{i}_{k}, b_{k}_{j}, {c_ij});\n"


generate_dot_kernels(
    x86_f16c_fp16_fp16_fp32(),
    [
        "dot,1x32x1",
        "dot,2x32x1",
        "dot,2x16x1",
        "dot,3x16x1",
        "dot,4x16x1",
        "dot,4x8x1",
        "dot,6x8x1",
        "dot,8x8x1",
    ],
)

generate_dot_kernels(
    x86_f16c_fma3_fp16_fp16_fp32(),
    [
        "dot,1x32x1",
        "dot,2x32x1",
        "dot,1x16x1",
        "dot,2x16x1",
        "dot,4x16x1",
        "dot,2x8x1",
        "dot,4x8x1",
        "dot,8x8x1",
    ],
)

generate_dot_kernels(
    x86_avx512_fp16_fp16_fp32(),
    [
        "dot,1x64x1",
        "dot,2x64x1",
        "dot,3x64x1",
        "dot,4x64x1",
        "dot,5x64x1",
        "dot,2x32x1",
        "dot,3x32x1",
        "dot,4x32x1",
        "dot,5x32x1",
        # The kernels which are commented out should be good, but for some
        # reason they don't perform well. They don't seem to spill, so keeping
        # them until we understand why are they slower.
        # "dot,6x32x1",
        # "dot,8x32x1",
        # "dot,10x32x1",
        # "dot,12x32x1",
        "dot,5x16x1",
        # "dot,16x16x1",
    ],
)
