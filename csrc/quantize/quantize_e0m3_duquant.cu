// ============================================================================
//  Fused DuQuant glue for the Omega E0M3 consumer path (see the .cuh).
//
//  Thread mapping (both kernels): one warp OWNS one 64-block and loops over
//  a tile of rows. The block's rotation matrix is staged once into shared
//  memory (8KB bf16) per warp — NOT re-read per row. v1 mapped one warp per
//  (row, block), which re-read R m times per layer (m*K*128B of L2 traffic;
//  ~2GB for a single K=16384 prefix layer at m=968) and REGRESSED the graph
//  baseline by ~67ms. Row-tiling restores the correct bandwidth ledger.
//
//  Per row: the 64 gathered inputs live one-per-lane in registers and are
//  broadcast with __shfl_sync; lanes read R from smem conflict-free pairs
//  (R[d][lane], R[d][lane+32]).
//
//  Rounding chain replicates the PyTorch reference (omega_e0m3_linear.py):
//    in : fp32 acc -> round bf16 -> [S1: divide by bf16(act_scale), round
//         bf16] -> (fp16 cast is exact) -> per-16 E0M3 quantize
//    out: fp32 acc -> round fp16 -> round bf16 -> [+ bias, fp32 math, round
//         bf16]
//  Device helpers are duplicated locally to stay additive (same convention
//  as quantize_e0m3_sfa.cu).
// ============================================================================
#include "quantize_e0m3_duquant.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(__CUDA_ARCH__)
#  include "cutlass/cutlass.h"
#  include "cutlass/detail/sm100_blockscaled_layout.hpp"
#  include "cute/tensor.hpp"
#  define FV_HAVE_CUTLASS 1
#else
#  define FV_HAVE_CUTLASS 0
#endif

namespace flash_rt {
namespace fp4 {

namespace {

// Rows processed per warp per launch. Larger tiles amortize the smem stage
// of R; smaller tiles give more blocks (occupancy) for tall inputs.
constexpr int kRowsPerWarp = 64;
constexpr int kWarpsPerBlock = 4;  // 128 threads, 32KB static smem

// Sign-magnitude uniform INT4: code = s|mmm, value = (s ? -1 : 1) * mmm.
// Identical to quantize_e0m3_sfa.cu.
__device__ __forceinline__ uint8_t fp32_to_e0m3(float x) {
    int mag = __float2int_rn(fabsf(x));
    if (mag > 7) mag = 7;
    uint8_t sign = (x < 0.f && mag > 0) ? 0x8u : 0x0u;
    return sign | static_cast<uint8_t>(mag);
}

__device__ __forceinline__ __nv_fp8_e4m3 quantize_ue4m3_e0m3(float x) {
    return __nv_fp8_e4m3(fmaxf(x, 0.f));
}

// IEEE-exact division/reciprocal, matching quantize_e0m3_sfa.cu (see its
// comment on --use_fast_math).
__device__ __forceinline__ float e0m3_scale_from_amax(float amax) {
    float desired = __fdiv_rn(amax, 7.f);
    if (desired < 1e-12f) desired = 1e-12f;
    return desired;
}

#if FV_HAVE_CUTLASS

using Cfg = cutlass::detail::Sm1xxBlockScaledConfig<16>;

template <class LayoutSF>
__global__ void kernel_quantize_e0m3_duquant_sfa(
    const __nv_bfloat16* __restrict__ src,
    const int* __restrict__ perm,
    const __nv_bfloat16* __restrict__ rot,   // [nb, 64, 64]
    const float* __restrict__ act_scale,     // [K] or nullptr (S0)
    uint8_t* __restrict__ dst_packed,
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout, int M, int K) {
    __shared__ __nv_bfloat16 sR[kWarpsPerBlock][64 * 64];
    const int nb = K >> 6;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int j = blockIdx.x * kWarpsPerBlock + warp_in_block;
    if (j >= nb) return;  // no cross-warp sync; safe to exit early

    // Stage this warp's rotation block into smem (coalesced), once per tile.
    const __nv_bfloat16* rblk = rot + static_cast<size_t>(j) * 4096;
    __nv_bfloat16* s = sR[warp_in_block];
    for (int i = lane; i < 4096; i += 32) s[i] = rblk[i];
    __syncwarp();

    const int base = j << 6;
    const int row0 = blockIdx.y * kRowsPerWarp;
    const int row1 = min(row0 + kRowsPerWarp, M);
    for (int row = row0; row < row1; ++row) {
        const __nv_bfloat16* srow = src + static_cast<size_t>(row) * K;
        // Gather permuted inputs; lane holds d = lane and d = lane + 32.
        const float in0 = __bfloat162float(srow[perm[base + lane]]);
        const float in1 = __bfloat162float(srow[perm[base + lane + 32]]);

        float acc0 = 0.f, acc1 = 0.f;
        #pragma unroll 4
        for (int d = 0; d < 64; ++d) {
            const float v = __shfl_sync(0xffffffffu, d < 32 ? in0 : in1, d & 31);
            acc0 = fmaf(v, __bfloat162float(s[(d << 6) + lane]), acc0);
            acc1 = fmaf(v, __bfloat162float(s[(d << 6) + lane + 32]), acc1);
        }

        // Replicate: bmm output is rounded to bf16, then (S1) divided by the
        // bf16-cast static scale with a bf16 rounding of the quotient.
        float o0 = __bfloat162float(__float2bfloat16(acc0));
        float o1 = __bfloat162float(__float2bfloat16(acc1));
        if (act_scale != nullptr) {
            const float s0 =
                __bfloat162float(__float2bfloat16(act_scale[base + lane]));
            const float s1 =
                __bfloat162float(__float2bfloat16(act_scale[base + lane + 32]));
            o0 = __bfloat162float(__float2bfloat16(o0 / s0));
            o1 = __bfloat162float(__float2bfloat16(o1 / s1));
        }

        // Per-16 amax groups: g0 = o0 lanes 0-15, g1 = o0 lanes 16-31,
        // g2 = o1 lanes 0-15, g3 = o1 lanes 16-31.
        float a0 = fabsf(o0), a1 = fabsf(o1);
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
            a0 = fmaxf(a0, __shfl_down_sync(0xffffffffu, a0, off, 16));
            a1 = fmaxf(a1, __shfl_down_sync(0xffffffffu, a1, off, 16));
        }
        // Every lane now holds its group's amax (identical IEEE-exact
        // results across the 16 lanes, so no extra broadcast is needed).
        const float desired0 = e0m3_scale_from_amax(a0);
        const float desired1 = e0m3_scale_from_amax(a1);
        const float scale_dq0 =
            static_cast<float>(quantize_ue4m3_e0m3(desired0));
        const float scale_dq1 =
            static_cast<float>(quantize_ue4m3_e0m3(desired1));
        const float inv0 = __frcp_rn(scale_dq0);
        const float inv1 = __frcp_rn(scale_dq1);

        // Scale writes: lane 0 -> g0 + g2, lane 16 -> g1 + g3.
        if (lane == 0 || lane == 16) {
            const int half = lane >> 4;  // 0 or 1
            __nv_fp8_e4m3 sq_even = quantize_ue4m3_e0m3(desired0);
            __nv_fp8_e4m3 sq_odd = quantize_ue4m3_e0m3(desired1);
            const int blk16 = j << 2;  // first 16-block of this 64-block
            dst_sfa[layout(row, (blk16 + half) * 16, 0)] =
                *reinterpret_cast<uint8_t*>(&sq_even);
            dst_sfa[layout(row, (blk16 + 2 + half) * 16, 0)] =
                *reinterpret_cast<uint8_t*>(&sq_odd);
        }

        // Pack: even lanes fetch the odd lane's code and write one byte.
        const uint32_t code0 = fp32_to_e0m3(o0 * inv0);
        const uint32_t code1 = fp32_to_e0m3(o1 * inv1);
        const uint32_t hi0 = __shfl_down_sync(0xffffffffu, code0, 1);
        const uint32_t hi1 = __shfl_down_sync(0xffffffffu, code1, 1);
        if ((lane & 1) == 0) {
            uint8_t* orow = dst_packed + static_cast<size_t>(row) * (K >> 1)
                            + (j << 5);
            orow[lane >> 1] = static_cast<uint8_t>(code0 | (hi0 << 4));
            orow[16 + (lane >> 1)] = static_cast<uint8_t>(code1 | (hi1 << 4));
        }
    }
}

#endif  // FV_HAVE_CUTLASS

// Output side: no CUTLASS dependency (no scale layout), always compiled.
__global__ void kernel_duquant_rotate_out(
    const __half* __restrict__ src,
    const __half* __restrict__ rot,          // [nb, 64, 64]
    const __nv_bfloat16* __restrict__ bias,  // [N] or nullptr
    __nv_bfloat16* __restrict__ dst,
    int M, int N) {
    __shared__ __half sR[kWarpsPerBlock][64 * 64];
    const int nb = N >> 6;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int j = blockIdx.x * kWarpsPerBlock + warp_in_block;
    if (j >= nb) return;

    const __half* rblk = rot + static_cast<size_t>(j) * 4096;
    __half* s = sR[warp_in_block];
    for (int i = lane; i < 4096; i += 32) s[i] = rblk[i];
    __syncwarp();

    const int base = j << 6;
    const int row0 = blockIdx.y * kRowsPerWarp;
    const int row1 = min(row0 + kRowsPerWarp, M);
    for (int row = row0; row < row1; ++row) {
        const __half* srow = src + static_cast<size_t>(row) * N;
        // No perm on the output side: straight coalesced reads.
        const float in0 = __half2float(srow[base + lane]);
        const float in1 = __half2float(srow[base + lane + 32]);

        float acc0 = 0.f, acc1 = 0.f;
        #pragma unroll 4
        for (int d = 0; d < 64; ++d) {
            const float v = __shfl_sync(0xffffffffu, d < 32 ? in0 : in1, d & 31);
            acc0 = fmaf(v, __half2float(s[(d << 6) + lane]), acc0);
            acc1 = fmaf(v, __half2float(s[(d << 6) + lane + 32]), acc1);
        }

        // Replicate: bmm output rounded to fp16, then cast to bf16, then
        // bias add in fp32 math rounded to bf16.
        float o0 = __half2float(__float2half(acc0));
        float o1 = __half2float(__float2half(acc1));
        __nv_bfloat16* drow = dst + static_cast<size_t>(row) * N;
        if (bias != nullptr) {
            o0 = __bfloat162float(__float2bfloat16(o0)) +
                 __bfloat162float(bias[base + lane]);
            o1 = __bfloat162float(__float2bfloat16(o1)) +
                 __bfloat162float(bias[base + lane + 32]);
        }
        drow[base + lane] = __float2bfloat16(o0);
        drow[base + lane + 32] = __float2bfloat16(o1);
    }
}

}  // namespace

int quantize_e0m3_duquant_sfa_bf16(
    const void* src_bf16, const void* perm_i32, const void* rot_bf16,
    const void* act_scale_fp32, void* dst_packed, void* dst_sfa,
    int M, int K, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
  if (K % 64 != 0) return -1;
  const int nb = K >> 6;
  const int threads = kWarpsPerBlock * 32;
  dim3 grid((nb + kWarpsPerBlock - 1) / kWarpsPerBlock,
            (M + kRowsPerWarp - 1) / kRowsPerWarp);

  auto shape = cute::make_shape(M, 1, K, 1);  // SFA (A operand) layout
  auto layout = Cfg::tile_atom_to_shape_SFA(shape);
  kernel_quantize_e0m3_duquant_sfa<<<grid, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(src_bf16),
      reinterpret_cast<const int*>(perm_i32),
      reinterpret_cast<const __nv_bfloat16*>(rot_bf16),
      reinterpret_cast<const float*>(act_scale_fp32),
      reinterpret_cast<uint8_t*>(dst_packed),
      reinterpret_cast<uint8_t*>(dst_sfa),
      layout, M, K);
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
#else
  (void)src_bf16; (void)perm_i32; (void)rot_bf16; (void)act_scale_fp32;
  (void)dst_packed; (void)dst_sfa; (void)M; (void)K; (void)stream;
  return -2;
#endif
}

int duquant_rotate_out_bf16(
    const void* src_fp16, const void* rot_fp16, const void* bias_bf16,
    void* dst_bf16, int M, int N, cudaStream_t stream) {
  if (N % 64 != 0) return -1;
  const int nb = N >> 6;
  const int threads = kWarpsPerBlock * 32;
  dim3 grid((nb + kWarpsPerBlock - 1) / kWarpsPerBlock,
            (M + kRowsPerWarp - 1) / kRowsPerWarp);
  kernel_duquant_rotate_out<<<grid, threads, 0, stream>>>(
      reinterpret_cast<const __half*>(src_fp16),
      reinterpret_cast<const __half*>(rot_fp16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      reinterpret_cast<__nv_bfloat16*>(dst_bf16),
      M, N);
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace fp4
}  // namespace flash_rt
