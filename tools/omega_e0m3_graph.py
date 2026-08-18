#!/usr/bin/env python3
"""CUDA-Graph the pi0.5 flow-matching denoise loop for E0M3 serving.

Captures the fixed 10-step Euler loop (action expert only) into a single
torch.cuda.CUDAGraph, FlashRT pi05_thor style: every buffer the graph
touches is preallocated once and only ever updated in place (copy_),
the Python step loop is unrolled at capture time, and each inference is
fill-static-buffers -> replay().

Graph boundary (what stays eager, per inference):
  preprocess -> embed_prefix (3x SigLIP + language) -> prefix prefill
  (KV cache built by transformers DynamicCache) -> sample noise.
The prefix KV is then copied into static per-layer buffers; the graph
reads those fixed addresses. Shapes are single-bucket: pi05 pads the
prompt to max_token_len=200, so prefix_len = 3*256 + 200 = 968 always.

Capture blockers removed relative to PI0Pytorch.sample_actions
(src/openpi/models_pytorch/pi0_pytorch.py):
  A1  `while time >= -dt/2` on a device tensor -> unrolled python loop;
      the time-dependent adaRMS conditioning is precomputed for all 10
      steps at setup (time grid is deterministic) and indexed per step.
  A2  embed_suffix's per-step `torch.tensor(att_masks, device=...)` H2D
      copy -> all masks/position ids are static buffers filled by copy_.
  B2  prefix DynamicCache grows/allocates per call -> static KV slabs
      wrapped in a DynamicCache shell (suffix path only reads them via
      cache[layer_idx] and get_seq_length()).

The torch.cat(prefix_KV, suffix_KV) per layer per step allocates inside
the graph's private memory pool — legal and address-stable across
replays; left as is.

Failure policy: any exception during setup/capture disables the graph
permanently and falls back to the original eager sample_actions.
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("omega_e0m3_graph")

ENV_FLAG = "OMEGA_E0M3_CUDA_GRAPH"


class GraphedDenoise:
    """Wraps a PI0Pytorch model; install() replaces model.sample_actions."""

    def __init__(self, model, num_steps: int = 10):
        self.model = model
        self.num_steps = num_steps
        self.graph: torch.cuda.CUDAGraph | None = None
        self.enabled = True  # flips False permanently on capture failure

    # ── public ──────────────────────────────────────────────────────
    def install(self) -> None:
        orig = self.model.sample_actions
        self._orig = orig

        @torch.no_grad()
        def patched(device, observation, noise=None, num_steps=10, **kw):
            bsize = observation.state.shape[0]
            if (not self.enabled or noise is not None
                    or num_steps != self.num_steps or bsize != 1):
                return orig(device, observation, noise=noise,
                            num_steps=num_steps, **kw)
            try:
                return self._run(device, observation)
            except Exception as e:
                log.exception("[OMEGA-E0M3] cuda graph failed; "
                              "falling back to eager sample_actions")
                print(f"[OMEGA-E0M3] cuda graph: DISABLED "
                      f"({type(e).__name__}: {e}), eager fallback",
                      flush=True)
                self.enabled = False
                return orig(device, observation, noise=noise,
                            num_steps=num_steps, **kw)

        self.model.sample_actions = patched
        print("[OMEGA-E0M3] cuda graph: installed "
              f"(num_steps={self.num_steps})", flush=True)

    # ── per-inference ───────────────────────────────────────────────
    def _run(self, device, observation) -> torch.Tensor:
        m = self.model
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        images, img_masks, lang_tokens, lang_masks, _state = \
            m._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = \
            m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        mask4d_prefix = m._prepare_attention_masks_4d(prefix_att_2d)
        m.paligemma_with_expert.paligemma.language_model.config \
            ._attn_implementation = "eager"  # noqa: SLF001

        _, pkv = m.paligemma_with_expert.forward(
            attention_mask=mask4d_prefix,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        if self.graph is None:
            self._setup_and_capture(prefix_pad_masks, pkv, device)

        # Fill static inputs (addresses baked into the graph).
        for l in range(len(self.k_static)):
            self.k_static[l].copy_(pkv.key_cache[l])
            self.v_static[l].copy_(pkv.value_cache[l])

        horizon = m.config.action_horizon
        prefix_pad_2d = prefix_pad_masks[:, None, :] \
            .expand(1, horizon, prefix_pad_masks.shape[1])
        full_2d = torch.cat([prefix_pad_2d, self.suffix_att_2d], dim=2)
        self.mask4d.copy_(m._prepare_attention_masks_4d(full_2d))

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        self.pos_ids.copy_(
            prefix_offsets + torch.cumsum(self.suffix_pad, dim=1) - 1)

        noise = m.sample_noise(
            (1, horizon, m.config.action_dim), device)
        self.x_t.copy_(noise)

        self.graph.replay()
        return self.x_t.clone()

    # ── one-time setup + capture ────────────────────────────────────
    def _setup_and_capture(self, prefix_pad_masks, pkv, device) -> None:
        m = self.model
        horizon = m.config.action_horizon
        adim = m.config.action_dim
        prefix_len = prefix_pad_masks.shape[1]
        n_layers = len(pkv.key_cache)

        # Static KV slabs + DynamicCache shell (suffix path reads
        # cache[layer_idx] -> (K, V) and get_seq_length() -> shape[-2]).
        from transformers.cache_utils import DynamicCache
        self.k_static = [pkv.key_cache[l].clone() for l in range(n_layers)]
        self.v_static = [pkv.value_cache[l].clone() for l in range(n_layers)]
        self.cache = DynamicCache()
        self.cache.key_cache = self.k_static
        self.cache.value_cache = self.v_static

        # Static inputs.
        self.x_t = torch.zeros(1, horizon, adim,
                               dtype=torch.float32, device=device)
        self.mask4d = torch.zeros(1, 1, horizon, prefix_len + horizon,
                                  dtype=torch.float32, device=device)
        self.pos_ids = torch.zeros(1, horizon,
                                   dtype=torch.long, device=device)

        # Constant suffix masks: pad = all ones; att = [1] + [0]*(h-1)
        # (from embed_suffix's pi05 branch, action tokens causal).
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        self.suffix_pad = torch.ones(1, horizon,
                                     dtype=torch.bool, device=device)
        att = torch.tensor([1] + [0] * (horizon - 1),
                           dtype=torch.bfloat16, device=device)
        self.suffix_att_2d = make_att_2d_masks(
            self.suffix_pad, att[None, :].expand(1, horizon))

        # Bake per-step adaRMS conditioning: the time grid is
        # deterministic (t = 1.0, 0.9, ..., 0.1), so run embed_suffix's
        # time branch once per step at setup and index at replay.
        dummy_x = torch.zeros(1, horizon, adim,
                              dtype=torch.float32, device=device)
        dummy_state = torch.zeros(1, m.config.action_dim,
                                  dtype=torch.float32, device=device)
        ad = []
        for s in range(self.num_steps):
            t = torch.tensor(1.0 - s / self.num_steps,
                             dtype=torch.float32, device=device)
            _, _, _, adc = m.embed_suffix(dummy_state, dummy_x,
                                          t.expand(1))
            ad.append(adc)
        self.adarms = torch.stack(ad)  # [steps, 1, expert_width]

        m.paligemma_with_expert.gemma_expert.model.config \
            ._attn_implementation = "eager"  # noqa: SLF001

        # Warmup on a side stream (cuBLAS/CUTLASS workspaces, allocator
        # pools must be settled before capture), then capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._loop()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._loop()
        torch.cuda.synchronize()
        print(f"[OMEGA-E0M3] cuda graph: captured "
              f"(prefix_len={prefix_len}, layers={n_layers}, "
              f"steps={self.num_steps})", flush=True)

    def _loop(self) -> None:
        """The 10-step denoise loop, unrolled into the graph."""
        m = self.model
        horizon = m.config.action_horizon
        dt = -1.0 / self.num_steps
        for s in range(self.num_steps):
            embs = m.action_in_proj(self.x_t)
            out, _ = m.paligemma_with_expert.forward(
                attention_mask=self.mask4d,
                position_ids=self.pos_ids,
                past_key_values=self.cache,
                inputs_embeds=[None, embs],
                use_cache=False,
                adarms_cond=[None, self.adarms[s]],
            )
            suffix_out = out[1][:, -horizon:].to(dtype=torch.float32)
            v_t = m.action_out_proj(suffix_out)
            self.x_t.add_(v_t, alpha=dt)


def install(model, num_steps: int = 10) -> GraphedDenoise | None:
    """Install the graphed sample_actions if $OMEGA_E0M3_CUDA_GRAPH=1."""
    if os.environ.get(ENV_FLAG, "0") in ("0", "false", "False"):
        print("[OMEGA-E0M3] cuda graph: disabled "
              f"(set {ENV_FLAG}=1 to enable)", flush=True)
        return None
    gd = GraphedDenoise(model, num_steps=num_steps)
    gd.install()
    return gd
