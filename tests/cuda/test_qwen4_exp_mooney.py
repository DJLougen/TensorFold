"""Mooney on CUDA (``families/qwen4_exp/cuda/mooney*.py``): the rotated 2-bit expert reader.

Always: the segmented rotation against a float64 Sylvester reference (signs first, normalized
Hadamard, one bf16 rounding), its row invariance, and the packed-value checks of the affine kernels
at the pack's widths (2-bit group-128 rotated experts, 8-bit group-32 dense tensors, fp16 meta).

With ``TENSORFOLD_MOONEY_FLASHNEXT=<a Mooney checkpoint>`` also: the real manifest loads, a real
layer's routed experts match a float64 reference of the same transform, and a model cut to
``TENSORFOLD_MOONEY_FLASHNEXT_LAYERS`` layers (default 4, plus the head and the MTP head) gives
one-row steps' logits bit for bit and MTP-drafted decoding emitting serial decoding's tokens.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen4_exp.cuda import mooney  # noqa: E402
from tensorfold.families.qwen4_exp import mooney as manifest_mod  # noqa: E402

DEV = "cuda"
MODEL = os.environ.get("TENSORFOLD_MOONEY_FLASHNEXT", "")
LAYERS = int(os.environ.get("TENSORFOLD_MOONEY_FLASHNEXT_LAYERS", "4"))
needs_model = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(),
                                 reason="set TENSORFOLD_MOONEY_FLASHNEXT to a Mooney checkpoint")


def sylvester(n: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def rotate_ref(x: np.ndarray, signs: np.ndarray, blocks: list[int]) -> np.ndarray:
    """float64 reference: per block, (1/sqrt(b)) H_b diag(signs_b) applied to x's columns."""

    out = np.zeros_like(x, dtype=np.float64)
    off = 0
    for b in blocks:
        out[:, off:off + b] = (sylvester(b) @ (signs[off:off + b] * x[:, off:off + b].T).T).T / math.sqrt(b)
        off += b
    return out


def bf16(x: np.ndarray) -> np.ndarray:
    return torch.tensor(x.astype(np.float32)).to(torch.bfloat16).float().numpy()


class Spec:
    """The smallest RotSpec duck-type Rotation needs."""

    def __init__(self, name: str, blocks: list[int]):
        self.name, self.blocks = name, tuple(blocks)

    @property
    def dim(self) -> int:
        return sum(self.blocks)

    def segments(self):
        out, at = [], 0
        for b in self.blocks:
            out.append((at, b))
            at += b
        return out


# -- the rotation -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dim,blocks", [(256, [256]), (512, [256, 128, 128]), (1024, [512, 512]),
                                        (2560, [1024, 1024, 512]), (640, [512, 128])])
def test_rotate_matches_fp64_sylvester(dim, blocks):
    g = torch.Generator().manual_seed(dim)
    x = (torch.randn(33, dim, generator=g) * 0.4).to(torch.bfloat16)
    signs = torch.where(torch.rand(dim, generator=g) < 0.5, -1.0, 1.0)
    rot = mooney.Rotation("w", Spec("w", blocks), signs.float().to(DEV))
    got = rot.apply(x.to(DEV)).float().cpu().numpy()
    ref = bf16(rotate_ref(x.float().numpy().astype(np.float64), signs.numpy(), blocks))
    # fp32 butterfly rounds at most one bf16 ulp off the fp64 reference where its fp32 sum differs
    assert np.abs(got - ref).max() <= np.abs(ref).max() * 2 ** -8 + 1e-9
    row_alone = rot.apply(x[7:8].contiguous().to(DEV)).float().cpu().numpy()
    assert np.array_equal(row_alone[0], got[7])          # a row's bits do not depend on the batch


def test_rotate_refuses_non_pow2():
    rot = mooney.Rotation("w", Spec("w", [384]), torch.ones(384, device=DEV))
    with pytest.raises(Exception):
        rot.apply(torch.zeros(2, 384, dtype=torch.bfloat16, device=DEV))


# -- packed-value checks ------------------------------------------------------------------------------------------
def pack_affine(w: np.ndarray, bits: int, group: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MLX affine packing of a float weight matrix (MLX's own round of q = (w - b)/s)."""

    n, k = w.shape
    per = 32 // bits
    g = k // group
    wr = w.reshape(n, g, group)
    lo, hi = wr.min(-1, keepdims=True), wr.max(-1, keepdims=True)
    s = np.where(hi > lo, (hi - lo) / ((1 << bits) - 1), 1.0).squeeze(-1).astype(np.float16)
    b = lo.squeeze(-1).astype(np.float16)
    q = np.clip(np.round((wr - lo) / s[..., None]), 0, (1 << bits) - 1).astype(np.int64)
    packed = np.zeros((n, k * bits // 32), np.uint32)
    for j in range(k):
        packed[:, j // per] |= q[:, j // group, j % group] << ((j % per) * bits)
    return packed.view(np.int32), s, b


def dequant_ref(words: np.ndarray, scales: np.ndarray, biases: np.ndarray, bits: int,
                group: int) -> np.ndarray:
    n = words.shape[0]
    k = words.shape[1] * 32 // bits
    q = np.zeros((n, k), np.float64)
    per = 32 // bits
    w = words.astype(np.int64) & 0xFFFFFFFF
    for j in range(k):
        q[:, j] = (w[:, j // per] >> ((j % per) * bits)) & ((1 << bits) - 1)
    return q * scales.astype(np.float64).repeat(group, -1) + biases.astype(np.float64).repeat(group, -1)


@pytest.mark.parametrize("bits,group", [(2, 128), (8, 32)])
def test_aff_matches_dequantized_fp16_meta(bits, group):
    g = torch.Generator().manual_seed(bits * 1000 + group)
    n, k = 96, 512
    w = (torch.randn(n, k, generator=g) * 0.05).numpy().astype(np.float32)
    words, s, b = pack_affine(w, bits, group)
    lin = mooney.Aff(torch.tensor(words, device=DEV), torch.tensor(s, dtype=torch.float16, device=DEV),
                     torch.tensor(b, dtype=torch.float16, device=DEV), bits, group, "mlx")
    x = torch.randn(6, k, generator=g).to(torch.bfloat16).to(DEV)
    got = lin(x).float().cpu().numpy()
    ref = (x.float().cpu().numpy().astype(np.float64)
           @ dequant_ref(words, s, b, bits, group).T)
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-2
    for rows in (1, 2, 5):
        part = lin(x[:rows].contiguous()).float().cpu().numpy()
        assert np.array_equal(part, got[:rows]), rows     # a row alone equals its row in the batch


# -- the manifest -------------------------------------------------------------------------------------------------
def test_manifest_refuses_bad(tmp_path):
    import json

    bad = tmp_path / "m"
    bad.mkdir()
    (bad / "mooney_rotation.json").write_text(json.dumps({
        "format": "lowbitflash.rot", "version": 99,
        "hadamard": {"kind": "sylvester_normalized"},
        "transform": {"order": "signs_first_then_hadamard"},
        "weights": {}, "inverses": {}}))
    assert manifest_mod.refusal(bad) is not None
    (bad / "mooney_rotation.json").write_text(json.dumps({
        "format": "lowbitflash.rot", "version": 1,
        "hadamard": {"kind": "sylvester_normalized"},
        "transform": {"order": "hadamard_first"},
        "weights": {"x": {}}, "inverses": {}}))
    assert manifest_mod.refusal(bad) is not None


# -- the real checkpoint ------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cut_model():
    from tensorfold.families.qwen4_exp.cuda import weights as W
    from tensorfold.families.qwen4_exp.cuda import mooney_load

    real = W.Config.read

    def cut(d):
        c = real(d)
        c.layers = LAYERS
        c.ple_layers = [i for i in c.ple_layers if i < LAYERS]
        return c

    W.Config.read = staticmethod(cut)
    try:
        w = mooney_load.load(MODEL, DEV, mtp=True, draft_vocab="default")
    finally:
        W.Config.read = real
    yield w
    del w
    torch.cuda.empty_cache()


@needs_model
def test_mooney_manifest_and_load(cut_model):
    """The pack's manifest parses against its index and a cut model's weights load onto the GPU."""

    from tensorfold.families.qwen4_exp import mooney as mm

    man = mm.manifest(MODEL)
    assert man.weights, "the manifest names no rotated weights"
    assert cut_model is not None


@needs_model
def test_cut_model_windows_equal_one_row_steps(cut_model):
    from tensorfold.families.qwen4_exp.cuda.decode import Engine
    from tensorfold.families.qwen4_exp.cuda.forward import commit, forward

    w = cut_model
    e = Engine(w, capacity=512, max_rows=max(WINDOWS), prefill_rows=128, graphs=False)
    toks = [int(t) for t in np.random.default_rng(3).integers(0, w.cfg.vocab, size=150)]
    e.reset()
    ref = []
    for t in toks:
        ref.append(forward(w, e.st, e.buf, [t])[:1].clone())
        commit(w, e.st, e.buf, 1, 1)
    ref = torch.cat(ref)
    for rows in WINDOWS:
        e.reset()
        got = []
        for at in range(0, len(toks), rows):
            chunk = toks[at:at + rows]
            got.append(forward(w, e.st, e.buf, chunk)[:len(chunk)].clone())
            commit(w, e.st, e.buf, len(chunk), len(chunk))
        assert torch.equal(torch.cat(got), ref), rows
    e.reset()
    forward(w, e.st, e.buf, toks[:10])
    commit(w, e.st, e.buf, 10, 4)
    assert torch.equal(forward(w, e.st, e.buf, toks[4:12])[:8], ref[4:12])


@needs_model
@pytest.mark.parametrize("graphs", [False, True])
def test_cut_model_mtp_drafts_emit_serial_tokens(cut_model, graphs):
    """Drafted decoding emits serial decoding's tokens (greedy ``None`` and seeded sampling)."""

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode

    w = cut_model
    assert w.mtp is not None
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=64, graphs=graphs)
    prompt = [int(t) for t in np.random.default_rng(4).integers(0, w.cfg.vocab, size=37)]
    for sampling in (None, Sampling(seed=1234, top_k=20, top_p=0.95)):
        s = serial_decode(e, prefill(e, prompt, sampling, mtp=False), 48, sampling)
        d = mtp_decode(e, prefill(e, prompt, sampling, mtp=True), 48, sampling, depth=6)
        assert s.tokens == d.tokens, sampling


@needs_model
def test_cut_model_prompts_ignore_chunking_and_resume_as_fresh(cut_model):
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill

    w = cut_model
    prompt = [int(t) for t in np.random.default_rng(5).integers(0, w.cfg.vocab, size=150)]
    runs = []
    for rows in (150, 64, 17):
        e = Engine(w, capacity=512, max_rows=8, prefill_rows=rows, graphs=False)
        first = prefill(e, prompt, None)
        runs.append((first, e.st.snapshot(), e.last_streams.clone()))
    for first, snap, tail in runs[1:]:
        assert first == runs[0][0]
        assert torch.equal(tail, runs[0][2])
        for key in ("rec", "conv", "ple_tail"):
            assert torch.equal(snap[key], runs[0][1][key]), key
