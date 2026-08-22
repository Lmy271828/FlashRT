#!/usr/bin/env python3
"""Serve openpi with Omega-QVLA linears running on FlashRT E0M3 kernels.

Installs the omega_e0m3_v1 consumer as a monkeypatch over gr00t's wrap
classes (GptqLinear; optionally DuQuantLinear), then execs Omega-QVLA's
openpi_inference_service.py with the CLI passed through unchanged. The
GR00T_GPTQ_* / GR00T_DUQUANT_* environment stays exactly as the Omega
hybrid server expects — the wrap flow runs as usual, but matched layers
are built as E0M3 consumers from the converted artifact instead of
fake-quant layers.

Environment:
  OMEGA_E0M3_PACK           path to the converted artifact (required)
  OMEGA_E0M3_PATCH_DUQUANT  1 to also patch DuQuantLinear (PaliGemma side);
                            default 0 = expert (GptqLinear) only
  OMEGA_E0M3_DISABLE_COMPILE  1 (default) to no-op torch.compile. The
                            consumer layers graph-break on the pybind
                            kernels, so compiled mode buys nothing but pays
                            recompile storms + empty cudagraph captures
                            (~25 min per episode). Eager + tcgen05 GEMM is
                            both faster and simpler on this path. Set 0 to
                            keep the checkpoint's compile mode.
  OMEGA_E0M3_CUDA_GRAPH  0 to disable capturing the 10-step denoise loop
                            into a single CUDA graph
                            (tools/omega_e0m3_graph.py); default 1
                            (validated: smoke 10/10, 50-ep 45/50 = 90.0%
                            ≈ eager 90.4%). Capture is lazy (first
                            inference) and falls back to eager on failure.
                            Auto-skipped when PI05_T_GRID is set (custom
                            grids take the eager branch in sample_actions).
  OMEGA_E0M3_ZERO_NOISE  1 to start sampling from zeros instead of
                            N(0, I). Required for MIP-trained 2-step
                            models (step 1 input is all-zeros by design);
                            default: 1 when PI05_T_GRID is set, else 0.
  OMEGA_SERVICE             service script path
                            (default /opt/omega/scripts/openpi_inference_service.py)

Example (inside the openpi docker container, after the usual env):
  export OMEGA_E0M3_PACK=/workspace/third_party/flashrt/pi05_long_e0m3.pt
  python -u /workspace/third_party/flashrt/tools/serve_omega_e0m3.py \
      --model_path /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
      --data_config pi05_libero --port 8000

MIP 2-step example (t*=0.9):
  export OMEGA_E0M3_PACK=/workspace/third_party/flashrt/pi05_mip2step_e0m3.pt
  export PI05_T_GRID="1.0:-1.0;0.1:-0.1"   # zero-noise is auto-enabled
  python -u /workspace/third_party/flashrt/tools/serve_omega_e0m3.py \
      --model_path <mip-qat checkpoint dir> --data_config pi05_libero --port 8000
"""

from __future__ import annotations

import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import omega_e0m3_linear as oel


def _disable_torch_compile() -> None:
    import torch

    def _noop_compile(model=None, **kwargs):
        if model is None:
            return lambda m: m
        return model

    torch.compile = _noop_compile  # type: ignore[assignment]
    print("[OMEGA-E0M3] torch.compile disabled (eager mode)", flush=True)


def _install_cuda_graph_hook() -> None:
    """Wrap create_trained_policy so the graphed denoise gets installed on
    the served policy's model. Policy.__init__ caches
    `self._sample_actions = model.sample_actions` at construction time, so
    after installing the monkeypatch we must rebind the cached reference.
    Capture itself is lazy (first inference) and therefore runs after the
    Omega wrap step has replaced the linears with E0M3 consumers."""
    if os.environ.get("PI05_T_GRID"):
        print("[OMEGA-E0M3] cuda graph: skipped (PI05_T_GRID custom grid "
              "takes the eager branch)", flush=True)
        return
    import openpi.policies.policy_config as policy_config

    orig = policy_config.create_trained_policy

    def wrapped(*args, **kwargs):
        policy = orig(*args, **kwargs)
        import omega_e0m3_graph as oeg
        gd = oeg.install(policy._model)  # noqa: SLF001
        if gd is not None:
            policy._sample_actions = \
                policy._model.sample_actions  # noqa: SLF001
        return policy

    policy_config.create_trained_policy = wrapped


def _env_flag(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0") not in ("0", "false", "False")


def _install_zero_noise_hook() -> None:
    """Start sampling from zeros (MIP 2-step models expect an all-zeros
    step-1 input). sample_actions looks up self.sample_noise at call time,
    so patching the instance attribute is enough — no rebinding needed."""
    import openpi.policies.policy_config as policy_config

    orig = policy_config.create_trained_policy

    def wrapped(*args, **kwargs):
        policy = orig(*args, **kwargs)
        import torch
        model = policy._model  # noqa: SLF001

        def zero_noise(shape, device):
            return torch.zeros(shape, device=device)

        model.sample_noise = zero_noise
        print("[OMEGA-E0M3] zero-noise sampling installed (MIP mode)", flush=True)
        return policy

    policy_config.create_trained_policy = wrapped


def main() -> None:
    service = os.environ.get(
        "OMEGA_SERVICE", "/opt/omega/scripts/openpi_inference_service.py")
    patch_duquant = os.environ.get("OMEGA_E0M3_PATCH_DUQUANT", "0") \
        not in ("0", "false", "False")
    if os.environ.get("OMEGA_E0M3_DISABLE_COMPILE", "1") \
            not in ("0", "false", "False"):
        _disable_torch_compile()
    oel.install(patch_duquant=patch_duquant)
    _install_cuda_graph_hook()
    if _env_flag("OMEGA_E0M3_ZERO_NOISE", default=bool(os.environ.get("PI05_T_GRID"))):
        _install_zero_noise_hook()
    runpy.run_path(service, run_name="__main__")


if __name__ == "__main__":
    main()
