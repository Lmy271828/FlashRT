// ============================================================================
//  FlashRT — fused DuQuant glue for the Omega E0M3 consumer path.
//
//  Two kernels replacing the per-linear PyTorch glue chain in
//  tools/omega_e0m3_linear.py:
//
//    quantize_e0m3_duquant_sfa_bf16  (input side)
//      bf16 [M,K] -> perm gather -> per-64 block rotation (DuQuant blocks)
//      -> optional S1 actnorm divide -> per-16 E0M3 dynamic quantize + SFA.
//      Replaces: index_select + bmm + reshape copy + fp16 cast + quantize.
//
//    duquant_rotate_out_bf16  (output side)
//      fp16 [M,N] GEMM output -> per-64 block rotation (r_out blocks)
//      -> bf16 cast -> optional bias add. Replaces: bmm + reshape copy +
//      bf16 cast + bias add.
//
//  Both replicate the PyTorch path's rounding chain exactly (fp32 accumulate,
//  intermediate bf16/fp16 roundings preserved); only the reduction ORDER of
//  the 64-wide dot products differs from cuBLAS (<= 1 ulp of the rounded
//  dtype). Numerics are gated by tools/check_omega_e0m3_layer.py and the
//  action-cos harness.
//
//  Additive: does NOT modify quantize_e0m3_sfa.* or any existing kernel.
// ============================================================================
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

// bf16 src [M, K] row-major + int32 perm [K] + bf16 rotation blocks
// [K/64, 64, 64] (+ optional fp32 act_scale [K], nullptr = S0) ->
// packed [M, K/2] E0M3 + SFA tile-interleaved UE4M3 scales (is_sfb=false
// layout; activations only). Returns 0 on success.
int quantize_e0m3_duquant_sfa_bf16(
    const void* src_bf16, const void* perm_i32, const void* rot_bf16,
    const void* act_scale_fp32, void* dst_packed, void* dst_sfa,
    int M, int K, cudaStream_t stream);

// fp16 src [M, N] row-major (GEMM output) + fp16 rotation blocks
// [N/64, 64, 64] + optional bf16 bias [N] (nullptr = no bias) ->
// bf16 dst [M, N] row-major. Returns 0 on success.
int duquant_rotate_out_bf16(
    const void* src_fp16, const void* rot_fp16, const void* bias_bf16,
    void* dst_bf16, int M, int N, cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
