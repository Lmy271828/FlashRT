#!/usr/bin/env python3
"""Consumer for omega_e0m3_v1 artifacts: drop-in Linear on FlashRT E0M3 kernels.

Replaces the fake-quant matmul inside Omega-QVLA's GptqLinear/DuQuantLinear
with the real SM110 path, keeping everything else identical:

    x2 = bmm(x[..., perm].view(M, K/64, 64), R_in_blocks)   # torch, as Omega
    A  = quantize_e0m3_dynamic_sfa_fp16(x2)                 # per-16 dynamic, S0
    y' = cutlass_fp4_gemm_e0m3w(A, packed_W, SFB)           # tcgen05 GEMM
    y  = bmm(y'.view(M, N/64, 64), R_out_blocks) + bias     # torch, as Omega

Weights come pre-converted from an Omega pack by
tools/convert_omega_pack_e0m3.py (S0, --fold none; see
docs/omega_pack_e0m3.md). Requires the compiled flash_rt_fp4 extension
(i.e. Thor). Activation dtype note: inputs are typically bf16 in openpi;
they are rotated in their own dtype (as Omega does) and cast to fp16 only
at the quantize-kernel boundary — well within fp16 range for activations.

Direct use:

    import omega_e0m3_linear as oel
    art = oel.load_artifact("pi05_long_e0m3.pt")
    layer = oel.OmegaE0M3Linear(base_linear, name, art).cuda()
    y = layer(x)

Monkeypatching an Omega runtime (before its wrap step runs):

    oel.install("pi05_long_e0m3.pt")   # patches gr00t GptqLinear/DuQuantLinear
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

ARTIFACT_ENV = "OMEGA_E0M3_PACK"


def load_artifact(path: str) -> Dict[str, Any]:
    art = torch.load(path, map_location="cpu", weights_only=True)
    fmt = art.get("format")
    if fmt != "omega_e0m3_v1":
        raise ValueError(f"{path}: expected format 'omega_e0m3_v1', got {fmt!r}")
    if art.get("fold") != "none":
        raise ValueError(f"{path}: consumer implements S0 only, got "
                         f"fold={art.get('fold')!r}")
    return art


class OmegaE0M3Linear(nn.Module):
    """nn.Linear drop-in running the Omega record on FlashRT E0M3 kernels."""

    def __init__(self, base: nn.Linear, name: str, artifact: Dict[str, Any]):
        super().__init__()
        if name not in artifact["weights"]:
            raise KeyError(f"layer '{name}' not in artifact")
        w = artifact["weights"][name]
        aux = artifact["aux"][name]

        self.name = name
        self.in_features = int(w["K"])
        self.out_features = int(w["N"])
        if base.in_features != self.in_features or \
                base.out_features != self.out_features:
            raise ValueError(
                f"{name}: base Linear({base.in_features}->"
                f"{base.out_features}) vs artifact "
                f"({self.in_features}->{self.out_features}) mismatch")
        self.bias = (nn.Parameter(base.bias.detach().clone())
                     if base.bias is not None else None)

        self.register_buffer("_packed", w["packed"], persistent=False)
        self.register_buffer("_sfb", w["sfb"], persistent=False)
        self.register_buffer("_perm", aux["duquant_rotation_perm"].long(),
                             persistent=False)
        self.register_buffer("_r_in", aux["duquant_rotation_blocks"],
                             persistent=False)
        self.register_buffer("_r_out", aux["duquant_rotation_out_blocks"],
                             persistent=False)

    def _rotate(self, x: torch.Tensor, perm: Optional[torch.Tensor],
                blocks: torch.Tensor) -> torch.Tensor:
        """x[M, D] -> bmm(x[:, perm].view(M, nb, B), blocks); matches Omega."""
        nb, b, _ = blocks.shape
        if perm is not None:
            x = x.index_select(dim=-1, index=perm)
        x = x.reshape(-1, nb, b)
        x = torch.bmm(x.transpose(0, 1).contiguous(),
                      blocks.to(dtype=x.dtype))
        return x.transpose(0, 1).contiguous().reshape(x.shape[0], nb * b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import flash_rt.flash_rt_fp4 as fvk_fp4

        orig_shape, in_dtype = x.shape, x.dtype
        x2 = x.reshape(-1, orig_shape[-1])
        x2 = self._rotate(x2, self._perm, self._r_in)
        x2 = x2.to(torch.float16).contiguous()

        m, k = x2.shape
        n = self.out_features
        dev = x2.device
        stream = torch.cuda.current_stream(dev).cuda_stream

        a_packed = torch.empty(m, k // 2, dtype=torch.uint8, device=dev)
        a_sfa = torch.zeros(fvk_fp4.sfa_size_bytes(m, k, False),
                            dtype=torch.uint8, device=dev)
        rc = fvk_fp4.quantize_e0m3_dynamic_sfa_fp16(
            x2.data_ptr(), a_packed.data_ptr(), a_sfa.data_ptr(),
            m, k, False, stream)
        if rc != 0:
            raise RuntimeError(f"{self.name}: A quantize rc={rc}")

        y = torch.empty(m, n, dtype=torch.float16, device=dev)
        rc = fvk_fp4.cutlass_fp4_gemm_e0m3w(
            a_packed.data_ptr(), a_sfa.data_ptr(),
            self._packed.data_ptr(), self._sfb.data_ptr(), y.data_ptr(),
            m, n, k, 1.0, 0.0, stream, 0)
        if rc != 0:
            raise RuntimeError(f"{self.name}: gemm rc={rc:#x}")

        y = self._rotate(y, None, self._r_out)
        y = y.to(in_dtype).reshape(*orig_shape[:-1], n)
        if self.bias is not None:
            y = y + self.bias.to(dtype=y.dtype)
        return y

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, format=omega_e0m3_v1(S0)")


# ────────────────────────────────────────────────────────────────────
# Omega runtime integration
# ────────────────────────────────────────────────────────────────────
def wrap_omega_e0m3(model: nn.Module, layer_names, artifact: Dict[str, Any],
                    verbose: bool = True) -> int:
    """Replace named nn.Linear modules with OmegaE0M3Linear (wrap_gptq shape)."""
    replaced = 0
    for name in layer_names:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        mod = getattr(parent, parts[-1])
        if not isinstance(mod, nn.Linear) or name not in artifact["weights"]:
            continue
        setattr(parent, parts[-1], OmegaE0M3Linear(mod, name, artifact))
        replaced += 1
        if verbose:
            print(f"[OMEGA-E0M3][REPLACED] {name}", flush=True)
    if verbose:
        print(f"[OMEGA-E0M3] Total layers replaced: {replaced}", flush=True)
    return replaced


def install(artifact_path: Optional[str] = None,
            patch_gptq: bool = True, patch_duquant: bool = False) -> int:
    """Monkeypatch gr00t's GptqLinear (and optionally DuQuantLinear) so the
    Omega wrap step builds E0M3 consumers from the artifact instead of
    fake-quant layers. Call BEFORE enable_gptq_if_configured /
    enable_duquant_if_configured. Layers missing from the artifact fall
    back to the original classes. Returns artifact layer count."""
    path = artifact_path or os.environ.get(ARTIFACT_ENV)
    if not path:
        raise ValueError(f"pass artifact_path or set ${ARTIFACT_ENV}")
    artifact = load_artifact(path)

    def make_factory(orig_cls):
        def factory(base, name, cfg, weight_bits=None):
            if name in artifact["weights"]:
                return OmegaE0M3Linear(base, name, artifact)
            return orig_cls(base, name=name, cfg=cfg, weight_bits=weight_bits)
        return factory

    patched = 0
    if patch_gptq:
        import gr00t.quantization.gptq_layers as gptq_mod
        gptq_mod.GptqLinear = make_factory(gptq_mod.GptqLinear)
        patched += 1
    if patch_duquant:
        import gr00t.quantization.duquant_layers as duq_mod
        duq_mod.DuQuantLinear = make_factory(duq_mod.DuQuantLinear)
        patched += 1
    print(f"[OMEGA-E0M3] installed: artifact={path} "
          f"({len(artifact['weights'])} layers), patched {patched} classes",
          flush=True)
    return len(artifact["weights"])
