# Omega-QVLA pack format and the E0M3 consumption contract

Status: recon complete, converter/harness in `tools/` (Milestone 1).
Scope: `packs_hf/pi05_long/quantized.pt` (4.8 GB, pi0.5 LIBERO-10 recipe
`paligemma=svdh+gptq, expert=svdh+rtn+perstep`). Other Omega packs share the
`dit_svdquant_v1` record format but were not inspected.

## 1. Container

Plain `torch.save` dict, loadable with `weights_only=True` (no custom
classes). 253 top-level keys:

- 126 expert records:
  `paligemma_with_expert.gemma_expert.model.layers.{0..17}.{self_attn.{q,k,v,o}_proj,mlp.{gate,up,down}_proj}`
- 126 PaliGemma records:
  `paligemma_with_expert.paligemma.model.language_model.layers.{0..17}.<same>`
- `__meta__`: `{"recipe": str, "suite": "10", "fresh": bool}`

Small projections (state_proj, action_in/out_proj, time_mlp) are
deliberately absent — they break under A4 and stay BF16 at runtime.

## 2. Record schema (`format == "dit_svdquant_v1"`)

| field | shape / dtype | meaning |
|---|---|---|
| `weight_res_q` | `(out, in)` fp16 | fake-quantized-then-dequantized weight, **already in the rotated + permuted domain** |
| `lowrank_A` / `lowrank_B` | `(out, 0)` / `(in, 0)` fp16 | SVDQuant low-rank branch; rank = 0 in this pack (INT4-only path) |
| `act_scale_table` | `(num_steps, in)` fp32 | per-denoise-step, per-channel activation scales. Expert: `num_steps = 10`; PaliGemma: `1` |
| `duquant_rotation_blocks` | `(in/64, 64, 64)` fp16 | block-diagonal input rotation R_in |
| `duquant_rotation_perm` | `(in,)` int64 | input-channel permutation (applied before R_in) |
| `duquant_rotation_out_blocks` | `(out/64, 64, 64)` fp16 | block-diagonal output rotation (restore) |
| `weight_bits` / `a_bits` | int | 4 / 4 (expert). PaliGemma records also say 4 but the runtime overrides activations to A8 |
| `in_features` / `out_features` | int | redundant with tensor shapes |
| `n_calib_*`, `act_percentile`, `gptq_damp_percent` | scalars | calibration provenance |

Notably absent (vs. a classic GPTQ pack): no packed int4 bitstream, no
`qweight`/`qzeros`/group scales — the weight survives only as dequantized
fp16 on the 4-bit grid (~8k unique values per tensor). No `smooth_scale`.

## 3. Consumer math (Omega `gr00t/quantization/gptq_layers.py`, verified)

```
x2 = bmm(x[..., perm].view(N, in/64, 64), R_in_blocks)   # input rotation, runtime
x_q = clamp(round(x2 / s_t), -8, 7) * s_t                # s_t = act_scale_table[step]
y'  = x_q @ W_res_q^T                                    # bf16-promoted accumulate
y   = bmm(y'.view(N, out/64, 64), R_out_blocks) + bias   # output rotation restore
```

PaliGemma (`duquant_layers.py`) is identical in structure with A8
activations and a single-row scale table.

Consequences for a FlashRT consumer:

- The input rotation **cannot** be folded into `weight_res_q`: fake-quant
  sits between rotation and GEMM. It must run on activations (torch bmm, or
  a prologue kernel). Same for the output restore.
- The rotation is an exact orthonormal transform, so it does not by itself
  affect GEMM fidelity; fidelity questions live entirely in the quantizers.
- `weight_res_q` being plain fp16 means the converter re-quantizes from
  fp16 — no GPTQ bitstream decoding needed.

## 4. Mapping to the FlashRT E0M3 contract

FlashRT SM110 path (`csrc/gemm/fp4/cutlass_fp4_gemm_e0m3w_sm100.cuh`,
bindings in `csrc/fp4_bindings.cpp`):

- Weights: fp16 `[N, K]` → `quantize_e0m3_dynamic_sfa_fp16(..., is_sfb=True)`
  → packed E0M3 `[N, K/2]` + SFB tile-interleaved UE4M3 (per-16, amax/7).
- Activations: same kernel with `is_sfb=False` → packed + SFA.
- GEMM: `cutlass_fp4_gemm_e0m3w(A, SFA, B, SFB, D, M, N, K, α, β, stream,
  a_format)` with `a_format=0` for E0M3 activations (1 = E2M1).
- Buffer sizing: `flash_rt_fp4.sfa_size_bytes(N, K, is_sfb)`; scale buffers
  must be zero-initialized (tile-interleave pads K to 64-element atoms;
  garbage padding decodes as UE4M3 NaN).

Grid differences vs. Omega fake-quant:

| | Omega A4 | FlashRT E0M3 |
|---|---|---|
| element grid | int `[-8, 7]` (asymmetric clamp) | sign-magnitude uniform `[-7, 7]` |
| scale | static calibrated, **per-channel** fp32 | dynamic amax/7, **per-16** UE4M3 |
| weight grid | int4 per-channel-group (already dequantized) | per-16 UE4M3 |

The scale-granularity mismatch (per-channel static table vs. per-16 dynamic)
is the one real fidelity risk. Two candidate strategies, both implemented in
`tools/check_omega_e0m3_layer.py`:

- **S0 (drop the table)**: `A = e0m3(x2)`, `B = e0m3(W)`. Loses all
  calibration information.
- **S1 (fold step-mean table into W, per-step residual into A)**:
  `A = e0m3(x2 / s_t)` per step, `B = e0m3(W · diag(s̄))` once, where
  `s̄ = mean_t(s_t)`. Exact for the mean step; residual error scales with
  the table's step-to-step spread (measured: std/mean ≈ 10% on expert
  layer-0 q_proj).

  Mathematically S1 relies on `Σ_k q_k s_k W_nk = Σ_k q_k (s_k W_nk)`:
  a per-K-column scale commutes into the weight. A true per-step fold would
  need 10 weight copies (unacceptable), hence the mean fold.

RHT (per-16 Hadamard, `use_rht=1` variants) is orthogonal to the DuQuant
rotation — `(x2·H)(W·H)^T = x2·W^T` — and can be ablated on top of either
strategy if per-block distributions remain problematic.

### Measured (emulation mode, synthetic activations calibrated to q999 = 7·s_t)

Per-token cosine vs. the unquantized-activation reference, 4 layers
(M = 256 tokens, K = 1024–4096):

| layer | omega vs fp | S0 vs fp | S1 vs fp |
|---|---|---|---|
| expert L0 q_proj (K=1024) | 0.9927 | 0.9834 | 0.9052 |
| expert L0 down_proj (K=4096) | 0.9928 | 0.9825 | 0.9799 |
| expert L11 o_proj (K=2048) | 0.9929 | 0.9825 | 0.9615 |
| paligemma L0 gate_proj (K=2048) | 0.9928 | 0.9810 | 0.9775 |

**S0 wins everywhere; S1 is never better and sometimes much worse.** The
table's per-channel scale spread (~4×) distorts the weight distribution
when folded (small-scale columns share a per-16 block scale with large ones
and quantize to few levels), while S0's per-token dynamic per-16 amax turns
out to be a *better* quantizer than Omega's static per-channel table — the
DuQuant rotation+perm has already whitened per-channel magnitudes, so the
table is only a second-order correction. Cost of dropping it: ~0.01
per-token cosine on every layer tested.

Decision: **the converter emits S0 (`--fold none`) as the production
format**; `--fold mean` is kept for ablation only. This also shrinks the
runtime story — no per-step scale dispatch is needed on the E0M3 path.

Caveats before treating this as final: (a) synthetic activations
(lognormal + outlier channels, calibrated only at the q999 point) — real
activation tails differ; (b) torch emulation approximates UE4M3 rounding
and MMA accumulation order; (c) single-layer, 4 instances. Confirm with
`--mode kernel` on Thor, then with captured real activations, then LIBERO.

## 5. Milestone-1 deliverables

- `tools/convert_omega_pack_e0m3.py` — offline pack → E0M3 converter
  (S0 weight emission + aux tensors: perm, R_in/R_out blocks,
  act_scale_table). Runs where `flash_rt_fp4` is built (Thor).
- `tools/check_omega_e0m3_layer.py` — single-layer cosine harness:
  Omega fake-quant reference vs. FlashRT E0M3 GEMM (S0/S1), plus a pure
  torch emulation mode that runs without the extension for pre-checks.
  Emulation results and the S0 decision are in §4.

Next verification steps, in order: `--mode kernel` on Thor (real tcgen05
GEMM vs. emulation) → captured real activations instead of synthetic →
LIBERO paired SR on a runtime wired to the converted pack (Milestone 2).

Deferred: SVDQuant low-rank epilogue (rank = 0 everywhere in this pack),
per-step weight tables (10× memory; also refuted by the S0 result),
per-step activation scale dispatch (refuted by the S0 result).

## 6. Accuracy context (pi0.5 LIBERO-10, 500 episodes)

This pack's full recipe (W4A4 expert + W4A8 PaliGemma, per-step scales)
scores 93.2% vs. BF16 baseline 91.6% (McNemar p = 0.32, no significant
difference) on the Omega PyTorch fake-quant path. The E0M3 migration target
is therefore "no measurable SR loss against an already lossless baseline" —
the single-layer cosine gates are the leading indicator, LIBERO the final
one.
