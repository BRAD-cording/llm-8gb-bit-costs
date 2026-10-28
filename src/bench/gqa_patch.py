"""
Work around PyTorch's refusal to fuse grouped-query attention on this stack.

Measured on the target machine (RTX 4060, Windows, torch 2.6.0+cu124,
transformers 5.20.2):

    q=[1,16,4096,128]  kv=[1,8,4096,128]
      mem_efficient, heads as-is   -> RuntimeError: No available kernel
      flash,         heads as-is   -> RuntimeError: No available kernel
      mem_efficient, kv expanded   -> peak   84 MB
      math,          kv expanded   -> peak 2693 MB

i.e. because Qwen3 uses GQA, every fused kernel is skipped and PyTorch falls back
to the math backend, which materialises a 16x4096x4096 fp32 score matrix. The
penalty is 32x memory and it is invisible: no error, no warning that survives
into a log, just a model that OOMs at 4k context on an 8 GB card.

`enable_gqa=True` does NOT fix it on this build (tested), so we expand the KV
heads explicitly and register the wrapper as an attention implementation.

Usage:
    from bench.gqa_patch import enable_gqa_sdpa
    enable_gqa_sdpa(model)
"""
from __future__ import annotations

import os
import sys

import torch

# make `quant.*` importable no matter which entry point loaded this module
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def enable_gqa_sdpa(model, kv_bits: int | None = None,
                    kv_residual: int = 0, kv_group: int = 0,
                    kv_official: bool = False) -> None:
    """Register a fused-compatible attention function.

    If ``kv_bits`` is given, the key/value tensors are fake-quantised inside the
    attention call.  This is the CORRECT place to measure cache quantisation:
    by the time attention is invoked, RoPE has already been applied to the keys
    (HF order is k_proj -> RoPE -> cache -> attention), so quantising here
    matches what a quantised cache stores.  Quantising the k_proj output instead
    measures a pre-RoPE quantity and overstates the damage -- an earlier version
    of this file made exactly that mistake.
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from quant.fakequant import qdq_asymmetric  # local import: keeps torch-only use light

    name = "gqa_sdpa" if kv_bits is None else (
        f"gqa_sdpa_kv{kv_bits}" if not (kv_residual or kv_group or kv_official)
        else f"gqa_sdpa_kv{kv_bits}_r{kv_residual}_g{kv_group}_o{int(kv_official)}")
    if name in ALL_ATTENTION_FUNCTIONS:
        model.config._attn_implementation = name
        return

    def make_fn(bits: int | None, residual: int = 0, group: int = 0,
                official: bool = False):
        def fn(module, query, key, value, attention_mask, **kwargs):
            if bits is not None:
                # KIVI convention, and getting the axes the right way round
                # matters: K has outliers that are stable ACROSS the sequence
                # for a given channel, so K gets one scale per channel (reduce
                # over the sequence axis).  V has no such structure, so it is
                # quantised per token (reduce over head_dim).  Swapping the two
                # destroys the model -- an earlier revision of this file did
                # exactly that and reported +17000% perplexity at 4 bits.
                # key: [b, kv_heads, seq, head_dim]
                if official:
                    from quant.kivi_official import quantise_kv_kivi_official
                    key, value = quantise_kv_kivi_official(
                        key, value, bits, 32, residual or 128)
                elif residual or group:
                    from quant.strong import quantise_kv_grouped, quantise_kv_kivi
                    if group:
                        key, value = quantise_kv_grouped(key, value, bits,
                                                         residual, group)
                    else:
                        key, value = quantise_kv_kivi(key, value, bits, residual)
                else:
                    key = qdq_asymmetric(key, bits, dim=-2)
                    value = qdq_asymmetric(value, bits, dim=-1)
            n_q, n_kv = query.shape[1], key.shape[1]
            if n_kv != n_q:
                rep = n_q // n_kv
                key = key.repeat_interleave(rep, dim=1)
                value = value.repeat_interleave(rep, dim=1)
            # transformers 4.51 passes enable_gqa=True down to SDPA; combined
            # with heads we have already materialised it expands them a second
            # time (16 -> 32) and SDPA raises a shape mismatch.
            kwargs.pop("enable_gqa", None)
            # sdpa_attention_forward repeats KV heads itself via
            # module.num_key_value_groups when the torch build lacks native GQA
            # in SDPA. We have already materialised them, so suppress the second
            # expansion for the duration of the call.
            groups = getattr(module, "num_key_value_groups", None)
            if groups is not None:
                module.num_key_value_groups = 1
            try:
                return sdpa_attention_forward(module, query, key, value,
                                              attention_mask, **kwargs)
            finally:
                if groups is not None:
                    module.num_key_value_groups = groups
        return fn

    ALL_ATTENTION_FUNCTIONS.register(
        name, make_fn(kv_bits, kv_residual, kv_group, kv_official))

    model.config._attn_implementation = name
    for sub in ("text_config", "decoder"):
        cfg = getattr(model.config, sub, None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = name
