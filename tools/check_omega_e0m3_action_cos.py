#!/usr/bin/env python3
"""End-to-end action-cosine harness for the Omega-QVLA pi0.5 serving paths.

Three modes compared on identical LIBERO observations with identical
per-observation noise (paired):

  bf16 : unquantized openpi PyTorch policy (reference)
  fake : Omega hybrid fake-quant — expert GPTQ pack (W4A4) + PaliGemma
         runtime DuQuant (RTN W4, A8 by default) — the arm-D recipe
  e0m3 : same pack consumed by FlashRT E0M3 kernels on BOTH sides
         (route A: OMEGA_E0M3_PATCH_DUQUANT=1 semantics, eager; the CUDA
         graph replays the same kernel sequence so it is not needed here)

Metrics per pair (fake/bf16, e0m3/bf16, e0m3/fake), matching the four
gates of tests/bench_pi05_decoder_fp4_e2e.py:

  raw_cosine / raw_min_sample_cosine   — the normalized action chunk
                                         (sample_actions output, pre-unnorm)
  action_cosine / action_min_sample_cosine — policy.infer()["actions"]

Plus mean policy infer_ms per mode (policy_timing), a usable eager-latency
signal for the same path.

The FlashRT obs fixture (/tmp/libero_obs_2v_n8.npz) is NOT reusable here:
its state is concat(joint_pos, gripper) while the policy path expects
concat(eef_pos, axisangle(eef_quat), gripper_qpos). Use --record-fixture
to capture a policy-path fixture on the LIBERO client host (no GPU
needed), then rsync it to Thor.

Usage:

  # 1. Record fixture (x86 host, examples/libero venv, LIBERO installed):
  PYTHONPATH=third_party/libero python tools/check_omega_e0m3_action_cos.py \
      --record-fixture /tmp/pi05_libero10_obs_n10.npz

  # 2. Thor, openpi container (transformers_replace already copied, PYTHONPATH
  #    has openpi src + /opt/omega + this repo root):
  python tools/check_omega_e0m3_action_cos.py \
      --checkpoint /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
      --pack /opt/omega/packs_hf/pi05_long/quantized.pt \
      --artifact /workspace/third_party/flashrt/pi05_long_e0m3.pt \
      --fixture /tmp/pi05_libero10_obs_n10.npz

  Parent mode spawns the three children sequentially (one GPU) and prints
  a metric table. Exit code 0 always unless --gate is passed, in which
  case the NVFP4-derived thresholds (action cos >= 0.999, action
  min-sample >= 0.995, raw >= 0.995/0.995) are applied to the
  e0m3-vs-bf16 and fake-vs-bf16 pairs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

GR00T_ENV = {
    # Same env the Thor hybrid server runs with (tools/start_e0m3_server.sh).
    "GR00T_GPTQ": "1",
    "GR00T_GPTQ_INCLUDE":
        r".*paligemma_with_expert\.gemma_expert\.model\.layers\.[0-9]+\..*"
        r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*",
    "GR00T_GPTQ_WBITS_DEFAULT": "4",
    "GR00T_GPTQ_ABITS": "4",
    "GR00T_GPTQ_MISSING": "fallback",
    "GR00T_DUQUANT_INCLUDE":
        r".*paligemma_with_expert\.paligemma\.model\.language_model\.layers\."
        r"[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*",
    "GR00T_DUQUANT_ROT_MODE": "svd_hadamard",
    "GR00T_DUQUANT_PERM_SCORE": "weight",
    "GR00T_DUQUANT_BLOCK": "64",
    "GR00T_DUQUANT_BLOCK_OUT": "64",
    "GR00T_DUQUANT_PERMUTE": "1",
    "GR00T_DUQUANT_ROW_ROT": "restore",
    "GR00T_DUQUANT_ACT_PCT": "99.9",
    "GR00T_DUQUANT_CALIB_STEPS": "32",
    "GR00T_DUQUANT_LS": "0.15",
}

GATE_THRESHOLDS = {
    "action_cosine": 0.999,
    "action_min_sample_cosine": 0.995,
    "raw_cosine": 0.995,
    "raw_min_sample_cosine": 0.995,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--record-fixture", metavar="PATH",
                   help="record a policy-path obs fixture and exit "
                        "(run on the LIBERO client host)")
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--samples-per-task", type=int, default=1)
    p.add_argument("--child-mode", choices=("bf16", "fake", "e0m3"))
    p.add_argument("--checkpoint", help="pi0.5 PyTorch checkpoint dir")
    p.add_argument("--data-config", default="pi05_libero")
    p.add_argument("--pack", help="Omega pack (fake/e0m3 modes)")
    p.add_argument("--artifact", help="converted omega_e0m3_v1 artifact "
                                      "(e0m3 mode)")
    p.add_argument("--fixture", help="policy-path obs npz from "
                                     "--record-fixture")
    p.add_argument("--output-dir")
    p.add_argument("--warmup", type=int, default=40,
                   help="warmup infers before measuring; must exceed "
                        "GR00T_DUQUANT_CALIB_STEPS (32) so fake mode's "
                        "online DuQuant calibration is frozen")
    p.add_argument("--seed", type=int, default=20260725)
    p.add_argument("--gate", action="store_true",
                   help="apply the NVFP4-derived cosine gates and fail "
                        "nonzero on violation")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────
# Fixture recording (LIBERO client host; mirrors examples/libero/main.py)
# ────────────────────────────────────────────────────────────────────
def _quat2axisangle(quat):
    """Copied from robosuite (same as examples/libero/main.py)."""
    import math

    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def record_fixture(args) -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools

    dummy_action = [0.0] * 6 + [-1.0]
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    bundle = {"n": 0}
    prompts, task_ids = [], []
    i = 0
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                                 camera_heights=256, camera_widths=256)
        env.seed(args.seed)
        picks = np.linspace(0, len(init_states) - 1,
                            args.samples_per_task).astype(int)
        for si in picks:
            env.reset()
            obs = env.set_init_state(init_states[si])
            for _ in range(10):
                obs, *_ = env.step(dummy_action)
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(
                obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, 224, 224))
            wrist = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist, 224, 224))
            state = np.concatenate(
                (obs["robot0_eef_pos"],
                 _quat2axisangle(obs["robot0_eef_quat"]),
                 obs["robot0_gripper_qpos"])).astype(np.float32)
            bundle[f"img_{i}"] = img
            bundle[f"wrist_{i}"] = wrist
            bundle[f"state_{i}"] = state
            prompts.append(str(task.language))
            task_ids.append(task_id)
            print(f"obs {i}: task {task_id} init_state {si} "
                  f"prompt={prompts[-1]!r}")
            i += 1
        env.close()
    bundle["n"] = i
    bundle["prompts"] = np.array(prompts)
    bundle["task_ids"] = np.array(task_ids, dtype=np.int64)
    np.savez(args.record_fixture, **bundle)
    print(f"saved {args.record_fixture} ({i} observations)")
    return 0


# ────────────────────────────────────────────────────────────────────
# Child modes (Thor, one policy per process)
# ────────────────────────────────────────────────────────────────────
def run_child(args) -> int:
    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if args.child_mode in ("fake", "e0m3"):
        for k, v in GR00T_ENV.items():
            os.environ.setdefault(k, v)
        os.environ.setdefault("GR00T_GPTQ_PATH", args.pack)
    if args.child_mode == "e0m3":
        os.environ.setdefault("OMEGA_E0M3_PACK", args.artifact)
        # The E0M3 consumers graph-break torch.compile (pybind kernels);
        # eager is the production mode on this path.
        def _noop_compile(model=None, **kwargs):
            return (lambda m: m) if model is None else model
        torch.compile = _noop_compile  # type: ignore[assignment]

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config(args.data_config)
    policy = _policy_config.create_trained_policy(
        train_config, args.checkpoint, pytorch_device="cuda")
    model = policy._model  # noqa: SLF001

    if args.child_mode in ("fake", "e0m3"):
        if args.child_mode == "e0m3":
            import omega_e0m3_linear as oel
            oel.install(args.artifact, patch_gptq=True, patch_duquant=True)
        from gr00t.quantization import (  # noqa: PLC0415
            enable_duquant_if_configured, enable_gptq_if_configured)
        if not enable_gptq_if_configured(model):
            raise RuntimeError("GR00T_GPTQ enabled but 0 layers wrapped")
        enable_duquant_if_configured(model)
        try:  # DuQuant builds buffers on CPU; re-sync (service does the same)
            model.to(next(model.parameters()).device)
        except StopIteration:
            pass

    # Capture the raw (normalized) action chunk alongside each infer.
    captured: dict[str, torch.Tensor] = {}
    orig_sample = policy._sample_actions  # noqa: SLF001

    def capturing(*a, **k):
        out = orig_sample(*a, **k)
        captured["raw"] = out.detach()
        return out

    policy._sample_actions = capturing  # noqa: SLF001

    data = np.load(args.fixture, allow_pickle=False)
    n = int(data["n"])
    prompts = data["prompts"]

    horizon, adim = model.config.action_horizon, model.config.action_dim
    noises = [np.random.default_rng(args.seed + i)
              .standard_normal((1, horizon, adim), dtype=np.float32)
              for i in range(n)]

    def obs(i):
        return {
            "observation/image": data[f"img_{i}"],
            "observation/wrist_image": data[f"wrist_{i}"],
            "observation/state": data[f"state_{i}"].astype(np.float32),
            "prompt": str(prompts[i]),
        }

    # Warmup: settles cudnn/cuBLAS workspaces AND fake mode's online
    # DuQuant calibration (first 32 forwards collect scales).
    for j in range(args.warmup):
        policy.infer(obs(j % n))

    raw_out, act_out, infer_ms = [], [], []
    for i in range(n):
        out = policy.infer(obs(i), noise=noises[i])
        raw_out.append(captured["raw"][0].float().cpu().numpy())
        act_out.append(out["actions"].astype(np.float64))
        infer_ms.append(float(out["policy_timing"]["infer_ms"]))

    raw = np.stack(raw_out).astype(np.float64)
    actions = np.stack(act_out)
    if not np.isfinite(raw).all() or not np.isfinite(actions).all():
        raise RuntimeError(f"{args.child_mode}: non-finite output")
    out_dir = Path(args.output_dir)
    np.savez(out_dir / f"{args.child_mode}_actions.npz",
             raw=raw, actions=actions)
    result = {
        "mode": args.child_mode,
        "n": n,
        "infer_ms_mean": float(np.mean(infer_ms)),
        "infer_ms_p50": float(np.median(infer_ms)),
    }
    print("__ACTION_COS_CHILD__ " + json.dumps(result, sort_keys=True),
          flush=True)
    return 0


# ────────────────────────────────────────────────────────────────────
# Parent: spawn children, compute the four cosine metrics per pair
# ────────────────────────────────────────────────────────────────────
def _cosines(a: np.ndarray, b: np.ndarray):
    flat_a = a.reshape(a.shape[0], -1)
    flat_b = b.reshape(b.shape[0], -1)
    per = [float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
           for x, y in zip(flat_a, flat_b)]
    glob = float(flat_a.reshape(-1) @ flat_b.reshape(-1) / (
        np.linalg.norm(flat_a) * np.linalg.norm(flat_b) + 1e-12))
    return glob, min(per)


def main() -> int:
    args = parse_args()
    if args.record_fixture:
        return record_fixture(args)
    if args.fixture is None:
        raise SystemExit("--fixture is required")
    if args.output_dir is None:
        raise SystemExit("--output-dir is required")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.child_mode is not None:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required")
        return run_child(args)

    if args.checkpoint is None or args.pack is None or args.artifact is None:
        raise SystemExit("parent mode requires --checkpoint/--pack/--artifact")

    import subprocess

    children = {}
    for mode in ("bf16", "fake", "e0m3"):
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--child-mode", mode,
               "--checkpoint", str(Path(args.checkpoint).resolve()),
               "--pack", str(Path(args.pack).resolve()),
               "--artifact", str(Path(args.artifact).resolve()),
               "--fixture", str(Path(args.fixture).resolve()),
               "--output-dir", str(out_dir.resolve()),
               "--data-config", args.data_config,
               "--warmup", str(args.warmup),
               "--seed", str(args.seed)]
        child = subprocess.run(cmd, check=False, capture_output=True,
                               text=True, env=os.environ.copy())
        (out_dir / f"{mode}.log").write_text(child.stdout + child.stderr)
        if child.returncode != 0:
            raise RuntimeError(
                f"{mode} child failed rc={child.returncode}; "
                f"log: {out_dir / f'{mode}.log'}")
        lines = [l.removeprefix("__ACTION_COS_CHILD__ ")
                 for l in child.stdout.splitlines()
                 if l.startswith("__ACTION_COS_CHILD__ ")]
        if len(lines) != 1:
            raise RuntimeError(f"expected one result from {mode} child")
        children[mode] = json.loads(lines[0])

    arrs = {m: np.load(out_dir / f"{m}_actions.npz") for m in children}
    pairs = {}
    for lhs, rhs in (("fake", "bf16"), ("e0m3", "bf16"), ("e0m3", "fake")):
        raw_cos, raw_min = _cosines(arrs[lhs]["raw"], arrs[rhs]["raw"])
        act_cos, act_min = _cosines(arrs[lhs]["actions"],
                                    arrs[rhs]["actions"])
        pairs[f"{lhs}_vs_{rhs}"] = {
            "raw_cosine": raw_cos,
            "raw_min_sample_cosine": raw_min,
            "action_cosine": act_cos,
            "action_min_sample_cosine": act_min,
            "action_max_abs": float(np.max(np.abs(
                arrs[lhs]["actions"] - arrs[rhs]["actions"]))),
        }

    gates = {}
    if args.gate:
        for pair in ("fake_vs_bf16", "e0m3_vs_bf16"):
            for metric, thr in GATE_THRESHOLDS.items():
                gates[f"{pair}.{metric}_at_least_{thr}"] = (
                    pairs[pair][metric] >= thr)

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "pack": str(Path(args.pack).resolve()),
        "artifact": str(Path(args.artifact).resolve()),
        "fixture": str(Path(args.fixture).resolve()),
        "seed": args.seed,
        "children": children,
        "pairs": pairs,
        "gates": gates,
        "passed": all(gates.values()) if gates else None,
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"{'pair':<14} {'action_cos':>10} {'action_min':>10} "
          f"{'raw_cos':>10} {'raw_min':>10} {'max_abs':>8}")
    for name, m in pairs.items():
        print(f"{name:<14} {m['action_cosine']:>10.5f} "
              f"{m['action_min_sample_cosine']:>10.5f} "
              f"{m['raw_cosine']:>10.5f} {m['raw_min_sample_cosine']:>10.5f} "
              f"{m['action_max_abs']:>8.4f}")
    for mode, c in children.items():
        print(f"{mode}: infer p50 {c['infer_ms_p50']:.1f} ms "
              f"(mean {c['infer_ms_mean']:.1f})")
    if gates:
        print("GATES:", "PASS" if result["passed"] else "FAIL")
    print(f"Artifacts: {out_dir}")
    return 0 if (not gates or result["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
