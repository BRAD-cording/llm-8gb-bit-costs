"""
Compatibility shim: run gptqmodel 4.2.5 against transformers 5.x.

gptqmodel 4.2.5 imports a handful of names from `transformers.modeling_utils`
that moved in transformers 5.x. The rest of this project needs the 5.x line
(Qwen3 support on the installed torch), so rather than downgrade transformers we
re-expose the moved names on the module gptqmodel looks at.

Usage: import this module *before* importing gptqmodel.

    import tools.gptq_compat  # noqa: F401
    from gptqmodel import GPTQModel, QuantizeConfig

Every alias below was found by running the import and reading the ImportError;
the list is exhaustive for the import path, not for the whole library. If a
future call fails with another missing name, add it here rather than downgrading
transformers.
"""
from __future__ import annotations


def apply() -> list[str]:
    import os
    applied: list[str] = []

    # Windows has no usable Triton. gptqmodel calls torch.compile in its
    # quantisation loop, torch._dynamo then fails to find Triton and raises
    # instead of falling back. Disabling dynamo keeps the eager path, which is
    # the same arithmetic --- only slower.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    try:
        import torch
        torch._dynamo.config.disable = True
        torch._dynamo.config.suppress_errors = True
        applied.append("torch._dynamo disabled")
    except Exception:
        pass
    import transformers.modeling_utils as mu

    try:
        import transformers.initialization as init
    except Exception:  # pragma: no cover - older transformers
        init = None

    if not hasattr(mu, "no_init_weights") and init is not None and hasattr(init, "no_init_weights"):
        mu.no_init_weights = init.no_init_weights
        applied.append("modeling_utils.no_init_weights")

    return applied


_APPLIED = apply()

if __name__ == "__main__":
    print("aliases applied:", _APPLIED or "none")
