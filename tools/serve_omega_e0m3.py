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
  OMEGA_SERVICE             service script path
                            (default /opt/omega/scripts/openpi_inference_service.py)

Example (inside the openpi docker container, after the usual env):
  export OMEGA_E0M3_PACK=/workspace/third_party/flashrt/pi05_long_e0m3.pt
  python -u /workspace/third_party/flashrt/tools/serve_omega_e0m3.py \
      --model_path /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch \
      --data_config pi05_libero --port 8000
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


def main() -> None:
    service = os.environ.get(
        "OMEGA_SERVICE", "/opt/omega/scripts/openpi_inference_service.py")
    patch_duquant = os.environ.get("OMEGA_E0M3_PATCH_DUQUANT", "0") \
        not in ("0", "false", "False")
    if os.environ.get("OMEGA_E0M3_DISABLE_COMPILE", "1") \
            not in ("0", "false", "False"):
        _disable_torch_compile()
    oel.install(patch_duquant=patch_duquant)
    runpy.run_path(service, run_name="__main__")


if __name__ == "__main__":
    main()
