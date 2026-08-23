#!/usr/bin/env python3
"""FA4 attention shim for the omega E0M3 serving path (eager-only).

Swaps ``modeling_gemma.eager_attention_forward`` for FlashRT's vendored
FA4 kernels (``flashrt_fa4``, SM100-class incl. Thor's SM110; head_dim
256 runs on the dedicated 2-CTA kernel kept there for pi0.5). Both the
PaliGemma LM and the action expert read that module-global at call time
(modeling_gemma.py:315), and the pi0.5 path pins
``config._attn_implementation = "eager"`` at runtime — so patching the
function (not the registry) is the hook that survives.

Mask semantics: pi0.5's additive 4D masks carry exactly one piece of
information — which key columns are valid. Prefill: the non-pad prefix
(right-padded language block). Denoise: valid prefix + the 10-token
suffix block, which is NOT causal (embed_suffix's ar list [1,0,...]
cumsums to a constant, i.e. a bidirectional block). FA4 takes no
additive mask tensor, so the shim compacts fully-masked key columns out
and runs dense non-causal attention. Fully-masked pad *query* rows get
valid-key attention instead of the eager uniform-garbage output; both
are unread downstream.

Eager-only (M3a): the mask->indices reduction is data-dependent on the
host and breaks CUDA-graph capture. Graphed FA4 needs per-length
capture with compaction baked per prompt length (M3b); the bench forces
--eager when --fa4 is passed.

Container requirements (thor-fa4 extra): nvidia-cutlass-dsl==4.5.1 +
quack-kernels==0.4.1. CUTE_DSL_ARCH=sm_101a must be set before the DSL's
first import (the sm_110a default path hits an NVVM chip-string bug);
set here via setdefault ahead of the lazy vendor load.
"""

from __future__ import annotations

import os

_INSTALLED = False


def _load_fa4():
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_101a")
    os.environ.setdefault("FLASH_ATTENTION_ARCH", "sm_100a")
    try:
        from flash_rt.hardware.thor.fa4_backend import fa4_func, status
    except ImportError as e:
        raise RuntimeError(
            "omega_fa4: flash_rt fa4_backend not importable — run inside the "
            "openpi container with the flashrt repo on PYTHONPATH"
        ) from e
    fn = fa4_func()
    if fn is None:
        raise RuntimeError(
            "omega_fa4: vendored flashrt_fa4 unavailable: " + status()
            + " — install the thor-fa4 extra in the container: "
            "nvidia-cutlass-dsl==4.5.1 quack-kernels==0.4.1"
        )
    return fn


def install() -> None:
    """Patch modeling_gemma.eager_attention_forward to FA4 (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    fn = _load_fa4()

    import transformers.models.gemma.modeling_gemma as modeling_gemma

    def fa4_attention_forward(module, query, key, value, attention_mask=None,
                              scaling=None, dropout=0.0, **kwargs):
        del module, dropout, kwargs
        if attention_mask is not None:
            mask = attention_mask[:, :, :, : key.shape[-2]]
            row = mask[:, :, :1, :]  # (B, 1, 1, Tk)
            # pi0.5 masks have uniform key visibility across query rows
            # (prefill: valid prefix; denoise: valid prefix + suffix
            # block). Anything else is out of contract — refuse loudly.
            if not bool((mask == row).all()):
                raise RuntimeError(
                    "omega_fa4: attention mask rows differ per query; "
                    "key-compaction requires uniform visibility")
            idx = (row[0, 0, 0] == 0).nonzero(as_tuple=True)[0]
            key = key.index_select(2, idx)
            value = value.index_select(2, idx)
        # transformers layout (B,H,T,D) -> FA4 layout (B,T,H,D); only the
        # last dim must be contiguous, transposed views qualify. GQA is
        # native to the hd256 kernel — pass K/V unexpanded.
        out = fn(query.transpose(1, 2), key.transpose(1, 2),
                 value.transpose(1, 2), softmax_scale=scaling, causal=False)
        if isinstance(out, tuple):
            out = out[0]
        return out, None  # (B,T,H,D), matches the eager return contract

    modeling_gemma.eager_attention_forward = fa4_attention_forward
    _INSTALLED = True
    print("[OMEGA-FA4] gemma eager_attention_forward -> FA4 "
          "(non-causal, compacted keys; eager-only, no graph capture)",
          flush=True)
