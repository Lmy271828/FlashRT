#!/usr/bin/env python3
"""P0 smoke: can OmegaE0M3Linear.forward be captured in a CUDA graph?

The E0M3 consumer calls two pybind kernels per forward
(quantize_e0m3_dynamic_sfa_fp16, cutlass_fp4_gemm_e0m3w). Graph capture
survives only if they are pure async launches with no hidden device
sync — that cannot be confirmed from Python, so this script just tries
it on real hardware (Thor).

Checks:
  1. capture layer(x) on a side stream after 3 warmups  (A4 gate)
  2. replay with new input data via copy_ into the static buffer
  3. cosine(replay_out, eager_out) on the same input (expect ~1.0)

Usage (on Thor, flashrt repo root, venv active):
  python tools/check_omega_e0m3_graph_smoke.py \
      --artifact ./pi05_long_e0m3.pt \
      [--layer paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj]

Exit code 0 = capturable; 1 = capture/replay/mismatch failure.
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn

import omega_e0m3_linear as oel

DEFAULT_LAYER = ("paligemma_with_expert.gemma_expert.model.layers.0."
                 "self_attn.q_proj")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--layer", default=DEFAULT_LAYER)
    ap.add_argument("--batch", type=int, default=16,
                    help="M dim of the test activation (10 tokens + slack)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); return 1

    art = oel.load_artifact(args.artifact)
    w = art["weights"][args.layer]
    k, n = int(w["K"]), int(w["N"])
    base = nn.Linear(k, n, bias=True, dtype=torch.bfloat16)
    layer = oel.OmegaE0M3Linear(base, args.layer, art).cuda()
    print(f"layer: {args.layer}  N={n} K={k}  M={args.batch}")

    x_static = torch.randn(args.batch, 1, k, dtype=torch.bfloat16,
                           device="cuda")

    # Eager reference on the same input.
    with torch.no_grad():
        y_ref = layer(x_static).float()

    # Warmup on a side stream, then capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad():
                layer(x_static)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                y_graph = layer(x_static)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"CAPTURE FAILED: {type(e).__name__}: {e}")
        return 1
    print("capture: OK")

    # Replay 1: same input -> must match eager reference.
    g.replay()
    torch.cuda.synchronize()
    cos1 = torch.nn.functional.cosine_similarity(
        y_ref.flatten(), y_graph.float().flatten(), dim=0).item()

    # Replay 2: new input via copy_ -> compare against eager on same data.
    x2 = torch.randn_like(x_static)
    x_static.copy_(x2)
    with torch.no_grad():
        y2_ref = layer(x2).float()
    g.replay()
    torch.cuda.synchronize()
    cos2 = torch.nn.functional.cosine_similarity(
        y2_ref.flatten(), y_graph.float().flatten(), dim=0).item()

    print(f"cosine replay-vs-eager: same-input {cos1:.6f}  "
          f"new-input {cos2:.6f}")
    ok = cos1 > 0.9999 and cos2 > 0.9999
    print("RESULT:", "PASS" if ok else "FAIL (numerical mismatch)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
