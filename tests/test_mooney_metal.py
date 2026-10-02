"""Mooney on Metal (``kernels/qwen/flash_next/v1/mooney.py``): the segmented rotation and the
per-projection-input expert matmuls, against a float64 Sylvester reference.

Synthetic triples only (a real Mooney pack's unit dimensions, not a checkpoint): 2-bit group-128
routed experts with fp16 scales/biases, 8-bit group-32 shared expert, signs-first normalized
Hadamard blocks of mixed sizes. Rows are checked alone and batched; the shared slot reads the plain
activation while routed pairs read the rotated one.
"""

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels.qwen.flash_next.v1 import mooney  # noqa: E402


def sylvester(n: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def rotate_ref(x: np.ndarray, signs: np.ndarray, blocks: list[int]) -> np.ndarray:
    out = np.zeros(x.shape, np.float32)
    off = 0
    for b in blocks:
        h = sylvester(b).astype(np.float32)
        out[:, off:off + b] = (h @ (signs[off:off + b] * x[:, off:off + b]).astype(np.float32).T).T
        out[:, off:off + b] /= np.float32(math.sqrt(b))
        off += b
    return out


def bf16(x: np.ndarray) -> np.ndarray:
    return np.asarray(mx.array(x.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32))


def pack_affine(w: np.ndarray, bits: int, group: int):
    n, k = w.shape
    g = k // group
    wr = w.reshape(n, g, group)
    lo, hi = wr.min(-1, keepdims=True), wr.max(-1, keepdims=True)
    s = np.where(hi > lo, (hi - lo) / ((1 << bits) - 1), 1.0).squeeze(-1)
    q = np.clip(np.round((wr - lo) / s[..., None]), 0, (1 << bits) - 1).astype(np.uint32)
    s, b = s.astype(np.float16), lo.squeeze(-1).astype(np.float16)
    per = 32 // bits
    packed = np.zeros((n, k * bits // 32), np.uint32)
    for j in range(k):
        packed[:, j // per] |= q[:, j // group, j % group] << ((j % per) * bits)
    return packed, s, b


def dequant(words: np.ndarray, s: np.ndarray, b: np.ndarray, bits: int, group: int) -> np.ndarray:
    shp = words.shape
    k = shp[-1] * 32 // bits
    per = 32 // bits
    w = words.astype(np.uint64)
    q = np.zeros((*shp[:-1], k), np.float64)
    for j in range(k):
        q[..., j] = (w[..., j // per] >> ((j % per) * bits)) & ((1 << bits) - 1)
    return q * s.astype(np.float64).repeat(group, -1) + b.astype(np.float64).repeat(group, -1)


class _Q:
    """A quantized-linear duck: packed uint32 words, fp16 scales/biases, bits/group_size."""

    def __init__(self, w, s, b, bits, gs):
        self.weight = mx.array(w)
        self.scales = mx.array(s)
        self.biases = mx.array(b)
        self.bits = bits
        self.group_size = gs


DIMS, WIDTH, EXPERTS, TOPK = 512, 256, 64, 2
GU_BLOCKS, D_BLOCKS = [256, 128, 128], [128, 128]


def _pack(seed):
    rng = np.random.default_rng(seed)
    sg = rng.choice([-1, 1], DIMS)
    su = rng.choice([-1, 1], DIMS)
    sd = rng.choice([-1, 1], WIDTH)
    rg = np.zeros((DIMS, DIMS))
    ru = np.zeros((DIMS, DIMS))
    rd = np.zeros((WIDTH, WIDTH))
    for R, signs, blocks in ((rg, sg, GU_BLOCKS), (ru, su, GU_BLOCKS), (rd, sd, D_BLOCKS)):
        off = 0
        for b in blocks:
            R[off:off + b, off:off + b] = sylvester(b) / math.sqrt(b) * signs[off:off + b][None, :]
            off += b
    wg = np.einsum("enk,kj->enj", rng.standard_normal((EXPERTS, WIDTH, DIMS)) * 0.05, rg.T)
    wu = np.einsum("enk,kj->enj", rng.standard_normal((EXPERTS, WIDTH, DIMS)) * 0.05, ru.T)
    wd = np.einsum("ekn,nj->ekj", rng.standard_normal((EXPERTS, DIMS, WIDTH)) * 0.05, rd.T)
    sg_w = pack_affine(rng.standard_normal((WIDTH, DIMS)) * 0.05, 8, 32)
    su_w = pack_affine(rng.standard_normal((WIDTH, DIMS)) * 0.05, 8, 32)
    sd_w = pack_affine(rng.standard_normal((DIMS, WIDTH)) * 0.05, 8, 32)
    pack_ = {}
    for name, arr in (("gate_proj", wg), ("up_proj", wu), ("down_proj", wd)):
        pack_[name] = pack_affine(arr.reshape(-1, arr.shape[-1]), 2, 128)
        pack_[name] = (pack_[name][0].reshape(*arr.shape[:2], -1), pack_[name][1].reshape(*arr.shape[:2], -1),
                       pack_[name][2].reshape(*arr.shape[:2], -1))
    return dict(sg=sg, su=su, sd=sd, rg=rg, ru=ru, rd=rd, pack=pack_, shared=(sg_w, su_w, sd_w))


@pytest.mark.parametrize("dim,blocks", [(512, [256, 128, 128]), (2560, [1024, 1024, 512]),
                                        (640, [512, 128]), (256, [256])])
def test_rotate_matches_fp64(dim, blocks):
    rng = np.random.default_rng(dim)
    signs = rng.choice([-1, 1], dim)
    x = rng.standard_normal((33, dim)).astype(np.float32)
    got = mooney.rotate(mx.array(x).astype(mx.bfloat16), mx.array(signs.astype(np.float32)),
                        [(o, b) for o, b in zip(np.cumsum([0] + blocks)[:-1], blocks)], dims=dim)
    got = np.asarray(got.astype(mx.float32))
    ref = bf16(rotate_ref(x.astype(np.float64), signs, blocks))
    # fp32 butterfly vs the fp64 reference: last-bit fp32 sums can flip the bf16 rounding by an
    # ulp, and near-zero outputs sit inside fp32 accumulation noise of the block's magnitude
    scale = np.abs(x).max() * math.sqrt(max(blocks)) * 2 ** -12
    ulp = np.abs(ref) * 2 ** -7 + np.spacing(np.abs(ref))
    assert (np.abs(got - ref) <= np.maximum(ulp * 2, scale)).all()
    alone = mooney.rotate(mx.array(x[7:8]).astype(mx.bfloat16), mx.array(signs.astype(np.float32)),
                          [(o, b) for o, b in zip(np.cumsum([0] + blocks)[:-1], blocks)], dims=dim)
    assert np.array_equal(np.asarray(alone.astype(mx.float32))[0], got[7])   # row alone == row in batch


def test_gateup_rotated_inputs_and_shared_plain():
    p = _pack(11)
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((5, DIMS)) * 0.3).astype(np.float32)
    logits = rng.standard_normal((5, EXPERTS)).astype(np.float32)
    segs = [(o, b) for o, b in zip(np.cumsum([0] + GU_BLOCKS)[:-1], GU_BLOCKS)]
    xg = mooney.rotate(mx.array(x).astype(mx.bfloat16), mx.array(p["sg"].astype(np.float32)), segs, dims=DIMS)
    xu = mooney.rotate(mx.array(x).astype(mx.bfloat16), mx.array(p["su"].astype(np.float32)), segs, dims=DIMS)
    gate = _Q(*p["pack"]["gate_proj"], 2, 128)
    up = _Q(*p["pack"]["up_proj"], 2, 128)
    shg = _Q(*p["shared"][0], 8, 32)
    shu = _Q(*p["shared"][1], 8, 32)
    act, picks, wts = mooney.mooney_gateup(mx.array(x).astype(mx.bfloat16), xg, xu, mx.array(logits),
                                           TOPK, EXPERTS, gate, up, shared=(shg, shu))
    mx.eval(act, picks, wts)
    act, picks = np.asarray(act.astype(mx.float32)), np.asarray(picks)
    g_ref = dequant(*p["pack"]["gate_proj"], 2, 128)
    u_ref = dequant(*p["pack"]["up_proj"], 2, 128)
    xg_np = rotate_ref(x.astype(np.float64), p["sg"], GU_BLOCKS)
    xu_np = rotate_ref(x.astype(np.float64), p["su"], GU_BLOCKS)
    for r in range(5):
        for k in range(TOPK):
            e = int(picks[r, k])
            g = xg_np[r] @ g_ref[e].T
            u = xu_np[r] @ u_ref[e].T
            ref = (g / (1 + np.exp(-g))) * u
            assert np.abs(act[r, k] - ref).max() < 0.1, (r, k)
    shg_ref = dequant(*[a[None] for a in p["shared"][0]], 8, 32)[0]
    shu_ref = dequant(*[a[None] for a in p["shared"][1]], 8, 32)[0]
    g = x.astype(np.float64) @ shg_ref.T                      # shared reads the plain input
    u = x.astype(np.float64) @ shu_ref.T
    ref = (g / (1 + np.exp(-g))) * u
    assert np.abs(act[:, TOPK] - ref).max() < 0.1


def test_down_rotated_routed_and_plain_shared():
    p = _pack(12)
    rng = np.random.default_rng(3)
    act = mx.array((rng.standard_normal((5, TOPK + 1, WIDTH)) * 0.3)).astype(mx.bfloat16)
    picks = mx.array(rng.integers(0, EXPERTS, (5, TOPK)).astype(np.uint32))
    segs = [(o, b) for o, b in zip(np.cumsum([0] + D_BLOCKS)[:-1], D_BLOCKS)]
    act_rot = mooney.rotate(act.reshape(-1, WIDTH), mx.array(p["sd"].astype(np.float32)), segs,
                            dims=WIDTH).reshape(act.shape)
    down = _Q(*p["pack"]["down_proj"], 2, 128)
    shd = _Q(*p["shared"][2], 8, 32)
    y = mooney.mooney_down_y(act_rot, act, picks, down, shd)
    mx.eval(y)
    y = np.asarray(y.astype(mx.float32))
    d_ref = dequant(*p["pack"]["down_proj"], 2, 128)
    shd_ref = dequant(*[a[None] for a in p["shared"][2]], 8, 32)[0]
    act_r = np.asarray(act_rot.astype(mx.float32)).astype(np.float64)
    act_np = np.asarray(act.astype(mx.float32)).astype(np.float64)
    for r in range(5):
        for k in range(TOPK + 1):
            ref = act_np[r, TOPK] @ shd_ref.T if k == TOPK else act_r[r, k] @ d_ref[int(picks[r, k])].T
            assert np.abs(y[r, k] - ref).max() < 0.1, (r, k)
