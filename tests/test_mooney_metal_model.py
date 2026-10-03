"""Mooney on Metal, cut model: a pack truncated by ``tools/mooney_cut.py`` loads on MLX, the
manifest's rotations attach to every switch_mlp, and a forward produces finite logits.

``TENSORFOLD_MOONEY_CUT=<dir from mooney_cut.py>`` gates the load. PLE stays on SSD (all rows preserved),
so warming the entire table cannot displace RAM on a small Mac. MTP and fused decode remain enabled.
The test also runs the model
without its manifest attached (module reclasses reverted) to prove the rotations change the result
— a Mooney pack's rotated experts give different bits than the same weights unrotated.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

MODEL = os.environ.get("TENSORFOLD_MOONEY_CUT", "")
needs_model = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(),
                                 reason="set TENSORFOLD_MOONEY_CUT to a tools/mooney_cut.py output")


def _inner(wrapper):
    """qwen4_exp.load's FlashNext wrapper -> the underlying MLX model (named_modules lives there)."""

    return wrapper.model


def _forward(inner, tokens):
    """model(inputs, cache) logits on the DEFAULT path (fused decode kept — it applies the
    manifest rotation through the mooney branch of FusedDecode._moe)."""

    cache = inner.make_cache()
    return np.asarray(inner(np.array([tokens]), cache).astype(mx.float32))


@needs_model
def test_cut_pack_loads_and_forwards():
    from tensorfold.families.qwen4_exp import load

    wrapper, tokenizer = load(Path(MODEL), ple_on_ssd=True)
    inner = _inner(wrapper)
    rotated = [m for _, m in inner.named_modules() if "mooney" in getattr(m, "__dict__", {})]
    man = json.loads((Path(MODEL) / "mooney_rotation.json").read_text())
    assert len(rotated) == len(man["weights"]) // 3, "not every manifest layer attached"
    logits = _forward(inner, [1, 2, 3])
    assert np.isfinite(logits).all(), "non-finite logits"
    assert logits.shape[-1] > 0


@needs_model
def test_rotation_is_applied_not_cosmetic():
    """The same weights with the manifest's rotations detached must give different logits
    (the rotated basis decoded without its transform is not the model)."""

    from tensorfold.families.qwen4_exp import load
    from tensorfold.families.qwen4_exp.model_layers import MooneySwitchGLU

    wrapper, _ = load(Path(MODEL), ple_on_ssd=True)
    inner = _inner(wrapper)
    with_rot = _forward(inner, [3, 1, 4])
    mods = [m for _, m in inner.named_modules() if isinstance(m, MooneySwitchGLU)]
    assert mods, "no rotated layers attached"
    from mlx_lm.models.switch_layers import SwitchGLU

    for m in mods:
        m.__dict__.pop("mooney", None)     # fused _moe reads this dict
        m.__class__ = SwitchGLU            # the module path reads the class; both = same weights, no transform
    without_rot = _forward(inner, [3, 1, 4])
    assert not np.allclose(with_rot, without_rot), "rotation made no difference — transform not applied"
