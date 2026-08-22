#!/usr/bin/env python3
r"""Offline converter: MIP-QAT pi0.5 checkpoint -> FlashRT E0M3 weights.

Sibling of convert_omega_pack_e0m3.py: instead of an Omega-QVLA
dit_svdquant_v1 pack, the input is a plain safetensors checkpoint produced by
scripts/train_mip_qat.py (openpi). The QAT-trained action-expert weights are
already near-4-bit representable; this tool re-quantizes them losslessly-ish
into the FlashRT SM110 E0M3 operand format (packed 4-bit [N, K/2] +
tile-interleaved UE4M3 SFB scales, per-16 amax/7) via the
`quantize_e0m3_dynamic_sfa_fp16` kernel.

Aux contract (consumer: tools/omega_e0m3_linear.py): identity DuQuant
rotations/permutation (QAT v1 runs without rotation), act_scale_table = ones
(S0 semantics — no activation-side calibration table), fold="none".

Scope: the 126 action-expert attention/MLP projections (same include regex as
GR00T_GPTQ_INCLUDE). All other layers (PaliGemma ViT/LLM, projections) stay
in bf16 and are served from the checkpoint itself.

Requires: CUDA + compiled flash_rt_fp4 extension (run on Thor).

Usage:
  python tools/convert_qat_ckpt_e0m3.py \
      --ckpt ~/checkpoints/mip_qat_run/5000/model.safetensors \
      --out pi05_mip2step_e0m3.pt
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import torch

OUTPUT_FORMAT = "omega_e0m3_v1"
SOURCE_FORMAT = "mip_qat_ckpt_v1"

EXPERT_INCLUDE = (
    r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))\.weight$"
)

# Identity aux must match the block size the consumer rotates with.
ROT_BLOCK = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True, help="model.safetensors from train_mip_qat.py")
    p.add_argument("--out", required=True, help="output .pt path")
    p.add_argument("--layer-regex", default="", help="convert only matching layers")
    p.add_argument("--keep-fp16", action="store_true",
                   help="also store the fp16 weight, for offline reference checks")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        print("error: CUDA is required (quantize kernels run on GPU)", file=sys.stderr)
        return 2
    try:
        import flash_rt.flash_rt_fp4 as fvk_fp4
    except ImportError:
        print("error: flash_rt_fp4 extension not importable — run this on a "
              "machine with FlashRT built (Thor)", file=sys.stderr)
        return 2

    from safetensors.torch import load_file

    sd = load_file(args.ckpt)
    rx = re.compile(EXPERT_INCLUDE)
    names = sorted(k for k in sd if rx.search(k))
    if args.layer_regex:
        extra = re.compile(args.layer_regex)
        names = [n for n in names if extra.search(n)]
    if not names:
        print("error: no expert layers matched", file=sys.stderr)
        return 2

    device = torch.device("cuda")
    eye = torch.eye(ROT_BLOCK, dtype=torch.float16)
    weights: dict = {}
    aux: dict = {}
    t0 = time.time()
    for i, key in enumerate(names):
        name = key[: -len(".weight")]
        w = sd[key].to(device=device, dtype=torch.float16).contiguous()
        n, k = w.shape
        if k % 16 != 0:
            print(f"skip {name}: K={k} not divisible by 16")
            continue
        if k % ROT_BLOCK != 0 or n % ROT_BLOCK != 0:
            print(f"skip {name}: N={n} K={k} not divisible by rotation block {ROT_BLOCK}")
            continue

        packed = torch.empty(n, k // 2, dtype=torch.uint8, device=device)
        sfb = torch.zeros(fvk_fp4.sfa_size_bytes(n, k, True), dtype=torch.uint8, device=device)
        rc = fvk_fp4.quantize_e0m3_dynamic_sfa_fp16(
            w.data_ptr(), packed.data_ptr(), sfb.data_ptr(), n, k, True, 0)
        if rc != 0:
            raise RuntimeError(f"quantize_e0m3_dynamic_sfa_fp16 failed on {name}: rc={rc}")

        entry = {"packed": packed.cpu(), "sfb": sfb.cpu(), "N": n, "K": k}
        if args.keep_fp16:
            entry["weight_fp16_folded"] = w.cpu()
        weights[name] = entry

        aux[name] = {
            # Identity input/output rotations + identity permutation.
            "duquant_rotation_blocks": eye.repeat(k // ROT_BLOCK, 1, 1),
            "duquant_rotation_out_blocks": eye.repeat(n // ROT_BLOCK, 1, 1),
            "duquant_rotation_perm": torch.arange(k, dtype=torch.int64),
            # S0: no activation calibration table.
            "act_scale_table": torch.ones(1, k, dtype=torch.float32),
            "weight_bits": 4,
            "a_bits": 4,
            "in_features": k,
            "out_features": n,
            "rank": 0,
            "fold": "none",
        }

        if (i + 1) % 21 == 0 or i + 1 == len(names):
            print(f"[{i + 1}/{len(names)}] {name}  N={n} K={k}  ({time.time() - t0:.1f}s)")

    torch.cuda.synchronize()
    out = {
        "format": OUTPUT_FORMAT,
        "source_pack_meta": {"format": SOURCE_FORMAT, "ckpt": args.ckpt},
        "fold": "none",
        "weights": weights,
        "aux": aux,
    }
    torch.save(out, args.out)
    print(f"wrote {args.out}: {len(weights)} layers, fold=none, {time.time() - t0:.1f}s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
