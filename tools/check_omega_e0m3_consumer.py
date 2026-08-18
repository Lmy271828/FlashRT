#!/usr/bin/env python3
"""Unit check: OmegaE0M3Linear (FlashRT kernels) vs. a reference.

Two reference modes (--reference):

  gptq : Omega's own GptqLinear (fake-quant, reads the original pack).
         Needs gr00t importable (PYTHONPATH to Omega-QVLA). Expected cos
         ≈ 0.986 (the S0-vs-omega residual from the single-layer harness).
  fp16 : plain fp16 matmul through the same DuQuant rotations, weight =
         the pack's `weight_res_q` (valid for --fold none artifacts).
         gr00t-free — with a fixture pack from
         tools/gen_omega_pack_fixture.py this mode is fully
         self-contained. Expected cos ≈ 0.99 (E0M3 quantization error).

Both modes feed identical synthetic activations (q999-calibrated to the
layer's scale table, bf16 like the server) and time both paths with CUDA
events.

Requires Thor (flash_rt_fp4 + CUDA). Run from the flashrt repo:

  PYTHONPATH=$PWD/tools python tools/check_omega_e0m3_consumer.py \
      --pack ~/lmy/Omega-QVLA/packs_hf/pi05_long/quantized.pt \
      --artifact ./pi05_long_e0m3.pt

  # self-contained fixture round-trip (no Omega-QVLA checkout needed):
  python tools/gen_omega_pack_fixture.py --out /tmp/fixture_pack.pt
  python tools/convert_omega_pack_e0m3.py --pack /tmp/fixture_pack.pt \
      --out /tmp/fixture_e0m3.pt --fold none
  PYTHONPATH=$PWD/tools python tools/check_omega_e0m3_consumer.py \
      --reference fp16 --pack /tmp/fixture_pack.pt \
      --artifact /tmp/fixture_e0m3.pt
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
    p.add_argument("--reference", choices=("gptq", "fp16"), default="gptq",
                   help="gptq = Omega GptqLinear (needs gr00t); "
                        "fp16 = plain fp16 matmul reference (gr00t-free)")
    p.add_argument("--layers", nargs="*", default=None,
                   help="layer names to check (default: built-in list; "
                        "missing layers are skipped with a note)")
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

    if args.reference == "gptq":
        from gr00t.quantization.gptq_layers import GptqConfig, GptqLinear
        from gr00t.quantization.dit_step_context import set_dit_quant_step
    else:
        import contextlib

        def set_dit_quant_step(step, total):
            return contextlib.nullcontext()

    artifact = oel.load_artifact(args.artifact)
    pack = torch.load(args.pack, map_location="cpu", weights_only=True)

    names = args.layers if args.layers else LAYERS
    for name in names:
        if name not in pack or name not in artifact["weights"]:
            print(f"skip {name}: not present in pack/artifact")
            continue
        rec = pack[name]
        s_t = rec["act_scale_table"][args.step].float()
        in_f, out_f = int(rec["in_features"]), int(rec["out_features"])

        torch.manual_seed(args.seed)
        base = nn.Linear(in_f, out_f, bias=True).bfloat16().cuda()
        with torch.no_grad():
            base.weight.normal_(0, 0.02)
            base.bias.normal_(0, 0.01)

        ours = oel.OmegaE0M3Linear(base, name, artifact).cuda()
        x = synth_x(args.tokens, in_f, s_t, args.seed)

        with torch.no_grad(), set_dit_quant_step(args.step, 10):
            if args.reference == "gptq":
                cfg = GptqConfig(path=args.pack, act_bits=4,
                                 missing="error")
                ref = GptqLinear(base, name=name, cfg=cfg).cuda()
                y_ref = ref(x).float()
                ref_label = "GptqLinear"
            else:
                # fp16 reference: same DuQuant rotations as the consumer,
                # plain fp16 matmul with the pack's weight_res_q.
                w_fp16 = rec["weight_res_q"].cuda()  # [N, K] fp16
                x2 = ours._rotate(x.reshape(-1, in_f),  # noqa: SLF001
                                  ours._perm, ours._r_in)  # noqa: SLF001
                x2 = x2.to(torch.float16)
                y = x2 @ w_fp16.T
                y = ours._rotate(y, None, ours._r_out)  # noqa: SLF001
                y = y.to(x.dtype).reshape(-1, out_f) + base.bias
                y_ref = y.float().reshape(args.tokens, out_f)
                ref_label = "fp16"
                ref = None
            y_ours = ours(x).float()

        per = torch.nn.functional.cosine_similarity(y_ref, y_ours, dim=-1)
        t_ours = time_it(lambda: ours(x), args.iters)
        print(f"{name}")
        print(f"  cos vs {ref_label}: mean {per.mean():.6f}  "
              f"min {per.min():.6f}")
        if ref is not None:
            t_ref = time_it(lambda: ref(x), args.iters)
            print(f"  latency: {ref_label} {t_ref:.3f} ms   "
                  f"E0M3 {t_ours:.3f} ms   speedup {t_ref / t_ours:.2f}x")
        else:
            print(f"  latency: E0M3 {t_ours:.3f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
