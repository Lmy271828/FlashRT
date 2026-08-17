#!/usr/bin/env python3
"""Unit check: OmegaE0M3Linear (FlashRT kernels) vs. Omega GptqLinear.

Builds one base nn.Linear per requested layer, wraps it with Omega's own
GptqLinear (fake-quant reference, reading the original pack) and with the
E0M3 consumer (reading the converted artifact), feeds identical synthetic
activations (q999-calibrated to the layer's scale table, bf16 like the
server), and reports per-token cosine between the two outputs — expected
≈ 0.986 (the S0-vs-omega residual from the single-layer harness). Also
times both paths with CUDA events for a first latency signal.

Requires Thor (flash_rt_fp4 + gr00t + CUDA). Run from the flashrt repo with
Omega-QVLA importable (PYTHONPATH):

  PYTHONPATH=$PWD/tools python tools/check_omega_e0m3_consumer.py \
      --pack ~/lmy/Omega-QVLA/packs_hf/pi05_long/quantized.pt \
      --artifact ./pi05_long_e0m3.pt
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn

import omega_e0m3_linear as oel

LAYERS = [
    "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj",
    "paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj",
    "paligemma_with_expert.gemma_expert.model.layers.11.self_attn.o_proj",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pack", required=True, help="original Omega quantized.pt")
    p.add_argument("--artifact", required=True, help="converted omega_e0m3_v1")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=100)
    return p.parse_args()


def synth_x(tokens: int, in_f: int, s_t: torch.Tensor,
            seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    gains = torch.exp(torch.randn(in_f, generator=g))
    idx = torch.randperm(in_f, generator=g)[: in_f // 128 + 1]
    gains[idx] *= 10.0
    x = (torch.randn(tokens, in_f, generator=g) * gains)
    q999 = torch.quantile(x.abs(), 0.999, dim=0).clamp_min(1e-8)
    x = x * (7.0 * s_t.cpu() / q999)
    return x.bfloat16().cuda()


def time_it(fn, iters: int) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("error: CUDA required", file=sys.stderr)
        return 2

    from gr00t.quantization.gptq_layers import GptqConfig, GptqLinear
    from gr00t.quantization.dit_step_context import set_dit_quant_step

    artifact = oel.load_artifact(args.artifact)
    pack = torch.load(args.pack, map_location="cpu", weights_only=True)

    for name in LAYERS:
        rec = pack[name]
        s_t = rec["act_scale_table"][args.step].float()
        in_f, out_f = int(rec["in_features"]), int(rec["out_features"])

        torch.manual_seed(args.seed)
        base = nn.Linear(in_f, out_f, bias=True).bfloat16().cuda()
        with torch.no_grad():
            base.weight.normal_(0, 0.02)
            base.bias.normal_(0, 0.01)

        cfg = GptqConfig(path=args.pack, act_bits=4, missing="error")
        ref = GptqLinear(base, name=name, cfg=cfg).cuda()
        ours = oel.OmegaE0M3Linear(base, name, artifact).cuda()

        x = synth_x(args.tokens, in_f, s_t, args.seed)
        with torch.no_grad(), set_dit_quant_step(args.step, 10):
            y_ref = ref(x).float()
            y_ours = ours(x).float()

        per = torch.nn.functional.cosine_similarity(y_ref, y_ours, dim=-1)
        t_ref = time_it(lambda: ref(x), args.iters)
        t_ours = time_it(lambda: ours(x), args.iters)
        print(f"{name}")
        print(f"  cos vs GptqLinear: mean {per.mean():.6f}  "
              f"min {per.min():.6f}")
        print(f"  latency: GptqLinear {t_ref:.3f} ms   "
              f"E0M3 {t_ours:.3f} ms   speedup {t_ref / t_ours:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
