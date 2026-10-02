"""Mooney packs keep Linear(d, 1) shared-gate rows as bare vectors; sanitize() restores the row dim."""

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.families.qwen4_exp.mtp import sanitize  # noqa: E402

GATE = "language_model.mtp.layers.0.mlp.shared_expert_gate.weight"


def test_sanitize_restores_the_singleton_gate_row_without_touching_values():
    raw = (mx.arange(6) - 2).astype(mx.bfloat16)
    out = sanitize({GATE: raw})
    got = out["layers.0.mlp.shared_expert_gate.weight"]
    assert got.shape == (1, 6) and got.dtype == mx.bfloat16
    assert bool(mx.array_equal(got.reshape(-1), raw).item())


def test_sanitize_keeps_an_already_two_dimensional_gate():
    raw = mx.arange(6).reshape(1, 6).astype(mx.float32)
    out = sanitize({GATE: raw})
    got = out["layers.0.mlp.shared_expert_gate.weight"]
    assert got.shape == (1, 6) and got.dtype == mx.float32
    assert bool(mx.array_equal(got, raw).item())


def test_sanitize_does_not_guess_at_a_gate_that_is_not_a_singleton():
    raw = mx.zeros((2, 6))                                   # strict loading must still refuse this
    out = sanitize({GATE: raw})
    assert out["layers.0.mlp.shared_expert_gate.weight"].shape == (2, 6)


def test_sanitize_strips_mtp_prefixes_and_ignores_the_rest():
    norm = mx.ones((8,))                                     # 1D is fine for every other tensor
    out = sanitize({"language_model.mtp.layers.0.input_layernorm.weight": norm,
                    "mtp.fc_embedding.weight": mx.zeros((4, 8)),
                    "language_model.model.layers.0.mlp.shared_expert_gate.weight": norm,
                    "unrelated.weight": mx.zeros((3,))})
    assert out["layers.0.input_layernorm.weight"].shape == (8,)
    assert out["fc_embedding.weight"].shape == (4, 8)
    assert sorted(out) == ["fc_embedding.weight", "layers.0.input_layernorm.weight"]
