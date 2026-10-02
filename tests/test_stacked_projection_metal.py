"""The GDN input stack takes raw dense linears (in_proj_b/a are unquantized BF16) alongside
quantized ones: ``decode._stacked`` used to read ``l.bits`` on every member and died with
``AttributeError: 'Linear' object has no attribute 'bits'`` inside ``FusedDecode.__init__``.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")

from tensorfold.families.qwen4_exp import decode


def _quantized(k, n, key, bits=4, group=32):
    linear = nn.QuantizedLinear(k, n, bias=False, group_size=group, bits=bits)
    linear.set_dtype(mx.bfloat16)
    return linear


def _dense(k, n, key):
    linear = nn.Linear(k, n, bias=False)
    linear.weight = (0.05 * mx.random.normal((n, k), key=mx.random.key(key))).astype(mx.bfloat16)
    return linear


def _refs(x, linears):
    """Serial projections before _stacked rewires member weights into the shared buffers."""

    refs = [decode.project(x, linear) for linear in linears]
    mx.eval(*refs)
    return refs


def test_stacked_mixes_raw_dense_and_quantized_linears():
    """The model's [qkv, z, b, a] set: quantized rows and raw rows each get one matmul,
    outputs come back in member order, and no member is requantized."""

    k = 256
    linears = [_quantized(k, 128, 11), _quantized(k, 64, 12), _dense(k, 16, 13), _dense(k, 8, 14)]
    x = mx.random.normal((5, k), key=mx.random.key(15)).astype(mx.bfloat16)
    expected = mx.concatenate(_refs(x, linears), axis=-1)
    proj, _ = decode._stacked(linears)
    out = proj(x)
    mx.eval(out)
    assert out.shape == (5, 128 + 64 + 16 + 8)
    assert mx.array_equal(out, expected)             # member order: qkv, z, b, a — a reorder fails here
    for dense in linears[2:]:
        assert not hasattr(dense, "scales") and not hasattr(dense, "bits")
        assert isinstance(dense, nn.Linear) and dense.weight.dtype == mx.bfloat16


def test_stacked_dense_linears_share_one_weight():
    """An all-dense set stacks like the quantized one: member weights become views of it."""

    k = 128
    linears = [_dense(k, 24, 21), _dense(k, 8, 22)]
    x = mx.random.normal((3, k), key=mx.random.key(23)).astype(mx.bfloat16)
    expected = mx.concatenate(_refs(x, linears), axis=-1)
    proj, cuts = decode._stacked(linears)
    assert isinstance(proj, nn.Linear) and not isinstance(proj, nn.QuantizedLinear)
    assert cuts == [24] and proj.__dict__["member_rows"] == [24, 8]
    assert mx.array_equal(linears[0].weight, proj.weight[:24])
    assert mx.array_equal(linears[1].weight, proj.weight[24:])
    assert mx.array_equal(proj(x), expected)


def test_stacked_quantized_linears_unchanged():
    """Same-bits rows still share one QuantizedLinear; a second bit width splits by width."""

    k = 256
    x = mx.random.normal((4, k), key=mx.random.key(33)).astype(mx.bfloat16)
    same = [_quantized(k, 64, 31), _quantized(k, 32, 32)]
    expected = mx.concatenate(_refs(x, same), axis=-1)
    proj, cuts = decode._stacked(same)
    assert isinstance(proj, nn.QuantizedLinear)
    assert cuts == [64] and mx.array_equal(proj(x), expected)
    mixed = [_quantized(k, 64, 34, bits=4), _quantized(k, 32, 35, bits=8), _quantized(k, 16, 36, bits=4)]
    expected = mx.concatenate(_refs(x, mixed), axis=-1)
    split, _ = decode._stacked(mixed)
    assert isinstance(split, decode._Split)
    assert mx.array_equal(split(x), expected)        # member order q, 8-bit, q restored across the split
