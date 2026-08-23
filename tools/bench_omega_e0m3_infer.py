#!/usr/bin/env python3
"""Per-inference latency bench for the Omega E0M3 serving path (Thor).

Measures wall-clock around the full `policy.infer()` call — the same
measurement basis as `deployment_scripts/pi05_inference.py` (arms
A/B/C: 137.2 / 85.8 / 49.9 ms): input transforms, H2D, tokenize, model
forward (CUDA-graph replays when enabled), D2H and output unnormalize
are all inside the timer; only network/serialization is excluded. The
inner `policy_timing.infer_ms` (model segment only) is reported
alongside as a diagnostic.

The graphed path is the production mode: `noise=None` on every call so
`omega_e0m3_graph`'s patched sample_actions takes the prefix+denoise
graph replays. `--eager` skips the graph install as a reference arm.

Usage (Thor, openpi container; transformers_replace copied, PYTHONPATH
has openpi src + /opt/omega + this repo root):

  python tools/bench_omega_e0m3_infer.py \
      --checkpoint /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
      --pack /opt/omega/packs_hf/pi05_long/quantized.pt \
      --artifact /workspace/third_party/flashrt/pi05_long_e0m3.pt \
      --fixture /workspace/pi05_libero10_obs_n10.npz

Warmup includes the one-time graph capture (minutes); --iters p50/p95
are the steady-state numbers. Every run prints whether the prefix and
denoise graphs are actually active — a latency claim without that
evidence is not a graph-mode claim.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pack", required=True, help="Omega pack (quantized.pt)")
    p.add_argument("--artifact", required=True,
                   help="converted omega_e0m3_v1 artifact")
    p.add_argument("--fixture", required=True,
                   help="policy-path obs npz from --record-fixture")
    p.add_argument("--data-config", default="pi05_libero")
    p.add_argument("--warmup", type=int, default=20,
                   help="includes the one-time graph capture")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--eager", action="store_true",
                   help="reference arm: do not install the CUDA graphs")
    p.add_argument("--fa4", action="store_true",
                   help="route Gemma attention through the vendored FA4 "
                        "kernels (eager-only, forces --eager)")
    p.add_argument("--save-actions", metavar="PATH",
                   help="save per-iter actions npy for cross-run numerics "
                        "gates (e.g. default vs --fa4)")
    p.add_argument("--output", help="optional json result path")
    return p.parse_args()


def _load_fixture(path: str):
    data = np.load(path, allow_pickle=False)
    n = int(data["n"])
    prompts = data["prompts"]

    def obs(i: int) -> dict:
        return {
            "observation/image": data[f"img_{i}"],
            "observation/wrist_image": data[f"wrist_{i}"],
            "observation/state": data[f"state_{i}"].astype(np.float32),
            "prompt": str(prompts[i]),
        }

    return n, obs


def main() -> int:
    args = parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from check_omega_e0m3_action_cos import GR00T_ENV  # noqa: PLC0415

    for k, v in GR00T_ENV.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("GR00T_GPTQ_PATH", args.pack)
    os.environ.setdefault("OMEGA_E0M3_PACK", args.artifact)
    # Route A: PaliGemma records also become E0M3 consumers.
    os.environ.setdefault("OMEGA_E0M3_PATCH_DUQUANT", "1")

    import torch

    # Same reason as the action-cos harness: compiled prefix attention
    # traces SDPA with an fp32 mask (dtype error); production is eager +
    # CUDA graphs.
    def _noop_compile(model=None, **kwargs):
        return (lambda m: m) if model is None else model
    torch.compile = _noop_compile  # type: ignore[assignment]

    import omega_e0m3_linear as oel
    oel.install(args.artifact, patch_gptq=True, patch_duquant=True)

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config(args.data_config)
    policy = _policy_config.create_trained_policy(
        train_config, args.checkpoint, pytorch_device="cuda")
    model = policy._model  # noqa: SLF001

    from gr00t.quantization import (  # noqa: PLC0415
        enable_duquant_if_configured, enable_gptq_if_configured)
    if not enable_gptq_if_configured(model):
        raise RuntimeError("GR00T_GPTQ enabled but 0 layers wrapped")
    enable_duquant_if_configured(model)
    try:  # DuQuant builds buffers on CPU; re-sync (service does the same)
        model.to(next(model.parameters()).device)
    except StopIteration:
        pass

    if args.fa4:
        import omega_fa4_attention  # noqa: PLC0415
        omega_fa4_attention.install()
        if not args.eager:
            print("[BENCH] --fa4 is capture-unsafe (host-side mask "
                  "reduction); forcing --eager. Graphed FA4 is M3b.",
                  flush=True)
            args.eager = True

    gd = None
    if not args.eager:
        import omega_e0m3_graph as oeg
        gd = oeg.install(model)
        if gd is not None:
            # Policy.__init__ cached the pre-patch sample_actions; rebind.
            policy._sample_actions = model.sample_actions  # noqa: SLF001

    n, obs = _load_fixture(args.fixture)

    for i in range(args.warmup):
        if i == 0:
            print("[BENCH] first infer includes graph capture; "
                  "expect minutes", flush=True)
        policy.infer(obs(i % n))
        if i == 0:
            print("[BENCH] first infer done", flush=True)

    wall_ms, inner_ms = [], []
    acts = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        out = policy.infer(obs(i % n))  # noise=None: graph path
        wall_ms.append((time.perf_counter() - t0) * 1e3)
        inner_ms.append(float(out["policy_timing"]["infer_ms"]))
        if args.save_actions:
            acts.append(np.asarray(out["actions"], dtype=np.float32))

    actions = np.asarray(out["actions"])
    if not np.isfinite(actions).all():
        raise RuntimeError("non-finite actions in the last iter")
    if args.save_actions:
        np.save(args.save_actions, np.stack(acts))

    def stats(xs):
        return {"p50": float(np.median(xs)), "p95": float(np.percentile(xs, 95)),
                "mean": float(np.mean(xs)), "min": float(np.min(xs)),
                "max": float(np.max(xs))}

    graph_state = "eager (--eager)"
    if gd is not None:
        denoise = "active" if (gd.enabled and gd.graph is not None) \
            else "DISABLED"
        prefix = ("active" if gd.prefix_graph is not None
                  else ("off" if not gd.prefix_enabled else "DISABLED"))
        graph_state = f"denoise graph: {denoise}, prefix graph: {prefix}"

    result = {
        "basis": "wall-clock full policy.infer, in-process (arm A basis); "
                 "inner = policy_timing.infer_ms (model segment)",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "artifact": str(Path(args.artifact).resolve()),
        "fixture": str(Path(args.fixture).resolve()),
        "fa4": bool(args.fa4),
        "graph_state": graph_state,
        "warmup": args.warmup,
        "iters": args.iters,
        "wall_ms": stats(wall_ms),
        "inner_ms": stats(inner_ms),
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
