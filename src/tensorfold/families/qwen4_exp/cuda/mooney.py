"""Mooney on CUDA: the manifest's segmented rotation in fp32 (one rounding to bf16), an affine/dense
matrix face on the shared kernels for its mixed formats, and grouped expert matmuls for its 2-bit
group-128 rotated experts.

The rotated expert matmul's input transform is llama.cpp's semantics (FORMAT.md v1): per contiguous
block ``x_b -> H_b * (signs_b * x_b)`` with ``H_b`` the normalized Sylvester Walsh-Hadamard, computed
in fp32 and rounded once to bf16, the same rounding the Prism rotate kernel documents. Gate and up
rotate the layer input separately (each projection's own signs); down rotates each routed pair's
activation.

The grouped kernels reuse ``tensorfold.cuda.experts``' Plan (members/items/counts, pairs sorted by
expert, 16-pair tiles) so a (item, column-tile) program touches one expert's weights and a pair's
bits never depend on the batch's other rows: drafted equals serial, one row equals batched, resumed
equals fresh. Pairs whose pick is the shared expert's slot are masked out; the shared expert runs on
the ordinary affine matmuls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl

from tensorfold.cuda.kernels import affine
from tensorfold.cuda.kernels.affine_kernels import codes as _codes


# ---------------------------------------------------------------------------
# segmented sign + normalized Hadamard: fp32 in, one rounding to bf16 out

@triton.jit
def _mooney_rotate(X, SIGNS, OUT, M, OFF: tl.constexpr, BS: tl.constexpr,
                   IN_STRIDE: tl.constexpr, OUT_STRIDE: tl.constexpr, LOG: tl.constexpr):
    """Rows of X[:, OFF:OFF + BS] -> OUT[:, OFF:OFF + BS]: fp32 signs then the normalized butterfly."""

    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.arange(0, BS)
    ok = rows[:, None] < M
    x = tl.load(X + rows[:, None] * IN_STRIDE + OFF + cols[None, :], mask=ok, other=0).to(tl.float32)
    s = tl.load(SIGNS + OFF + cols).to(tl.float32)
    v = x * s[None, :]
    for i in tl.static_range(LOG):
        h = 1 << i
        a = tl.reshape(v, (16, BS // (2 * h), 2, h))
        e, o = tl.split(tl.permute(a, (0, 1, 3, 2)))
        # join puts (sum, diff) in the pair slot; permute back so it lands at stride h, not stride 2
        v = tl.reshape(tl.permute(tl.join(e + o, e - o), (0, 1, 3, 2)), (16, BS))
    v = v * (1.0 / (BS ** 0.5))
    tl.store(OUT + rows[:, None] * OUT_STRIDE + OFF + cols[None, :], v.to(tl.bfloat16), mask=ok)


@dataclass
class Rotation:
    """One projection's rotation on device: fp32 signs plus its block plan."""

    name: str
    spec: Any                     # mooney.RotSpec
    signs: torch.Tensor           # [dim] fp32

    @classmethod
    def build(cls, spec, device) -> "Rotation":
        return cls(spec.name, spec, torch.tensor(spec.signs, dtype=torch.float32, device=device))

    @property
    def dim(self) -> int:
        return self.spec.dim

    def apply(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """x [M, >= dim] bf16 (any row stride) -> bf16; pass ``out`` of x's shape for reuse."""

        m, k = x.shape
        if k < self.dim:
            raise ValueError(f"{self.name}: x has {k} columns, the rotation spans {self.dim}")
        if out is None:
            out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
        elif tuple(out.shape) != tuple(x.shape):
            raise ValueError("rotation output must match x's shape")
        grid = (triton.cdiv(m, 16),)
        for off, size in self.spec.segments():
            _mooney_rotate[grid](x, self.signs, out, m, OFF=off, BS=size,
                                 IN_STRIDE=x.stride(0), OUT_STRIDE=out.stride(0),
                                 LOG=size.bit_length() - 1, num_warps=4)
        return out


# ---------------------------------------------------------------------------
# an affine/dense matrix on the shared generic kernels (any MLX bits x group, fp16/bf16/fp32 meta)

@dataclass
class Aff:
    """One linear: MLX-layout affine (``bits``/``gs``) or a dense weight, on the Q4 call surface."""

    weight: torch.Tensor          # int32 packed words [N, K*bits/32], or dense [N, K] floating-point
    scales: torch.Tensor | None = None
    biases: torch.Tensor | None = None
    bits: int = 0
    gs: int = 0
    layout: str = "mlx"
    kernel: str = "mooney-affine"

    @property
    def n(self) -> int:
        return int(self.weight.shape[0])

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return affine.matmul(x, self, out=out)

    def prefill(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return affine.matmul(x, self, out=out)

    def partials(self, x: torch.Tensor) -> torch.Tensor:
        """[1, M, N] fp32 for the hc reduction's slice-sum path."""

        return affine.matmul(x, self, f32=True).unsqueeze(0)


@dataclass
class AffStack:
    """Projections sharing one input, each an Aff, writing its own slice of the output."""

    parts: list
    splits: tuple[int, ...]
    kernel: str = "mooney-affine"

    @property
    def n(self) -> int:
        return sum(p.n for p in self.parts)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if out is None:
            out = torch.empty((x.shape[0], self.n), dtype=torch.bfloat16, device=x.device)
        at = 0
        for part, width in zip(self.parts, self.splits):
            affine.matmul(x, part, out=out[:, at:at + width])
            at += width
        return out

    def prefill(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return self.__call__(x, out)

    def partials(self, x: torch.Tensor) -> torch.Tensor:
        """[1, M, total] fp32: the parts' f32 results concatenated over the output dim."""

        return torch.cat([p.partials(x) for p in self.parts], dim=2)


def stack_of(parts: list) -> Any:
    """One Aff, or an AffStack when projections share the input."""
    return parts[0] if len(parts) == 1 else AffStack(parts, tuple(p.n for p in parts))


# ---------------------------------------------------------------------------
# grouped expert kernels over MLX-layout triples (expert axis first), one expert per item tile

@triton.jit
def _affine_group(X, W, S, B, item, rows, ok, cols,
                  N: tl.constexpr, K: tl.constexpr, BITS: tl.constexpr, GS: tl.constexpr,
                  XSTRIDE: tl.constexpr):
    """acc [16, 32] fp32: rows' x against item-expert weight rows ``cols``."""

    acc = tl.zeros((16, 32), tl.float32)
    NG: tl.constexpr = K // GS
    KW: tl.constexpr = K * BITS // 32
    n_vec = item * N + cols                        # this item's weight rows in the stacked [E, N, ..] triple
    for g in range(NG):
        k = g * GS + tl.arange(0, GS)
        q = _codes(W, n_vec[:, None], k[None, :], cols[:, None] < N, KW, BITS)
        x = tl.load(X + rows[:, None] * XSTRIDE + k[None, :], mask=ok[:, None], other=0)
        dot = tl.dot(x, tl.trans(q.to(tl.bfloat16)), input_precision="ieee")
        sc = tl.load(S + n_vec * NG + g, mask=cols < N, other=0).to(tl.float32)
        bi = tl.load(B + n_vec * NG + g, mask=cols < N, other=0).to(tl.float32)
        xs = tl.sum(x.to(tl.float32), 1)
        acc += dot * sc[None, :] + xs[:, None] * bi[None, :]
    return acc


@triton.jit
def _gateup(XG, XU, WG, SG, BG, WU, SU, BU, ITEMS, COUNTS, MEMBERS, OUT,
            N: tl.constexpr, K: tl.constexpr, EXPERTS: tl.constexpr,
            BITS: tl.constexpr, GS: tl.constexpr, SLOTS: tl.constexpr, XSTRIDE: tl.constexpr):
    """Item x 32 columns: bf16(silu(gate) * up) of the routed pairs; the shared slot is masked out."""

    it = tl.program_id(0)
    if it < tl.load(COUNTS):
        e = tl.load(ITEMS + it * 3)
        if e < EXPERTS:
            first = tl.load(ITEMS + it * 3 + 1)
            cnt = tl.load(ITEMS + it * 3 + 2)
            cols = tl.program_id(1) * 32 + tl.arange(0, 32)
            lanes = tl.arange(0, 16)
            pid = tl.load(MEMBERS + first + lanes, mask=lanes < cnt, other=-1)
            ok = (lanes < cnt) & (pid >= 0)
            rows = tl.maximum(pid, 0) // SLOTS
            accg = _affine_group(XG, WG, SG, BG, e, rows, ok, cols, N, K, BITS, GS, XSTRIDE)
            accu = _affine_group(XU, WU, SU, BU, e, rows, ok, cols, N, K, BITS, GS, XSTRIDE)
            gate = accg.to(tl.bfloat16).to(tl.float32)
            up = accu.to(tl.bfloat16).to(tl.float32)
            act = (gate / (1.0 + tl.exp(-gate)) * up).to(tl.bfloat16)
            tl.store(OUT + pid[:, None] * N + cols[None, :], act,
                     mask=ok[:, None] & (cols[None, :] < N))


@triton.jit
def _down(X, W, S, B, ITEMS, COUNTS, MEMBERS, OUT,
          N: tl.constexpr, K: tl.constexpr, EXPERTS: tl.constexpr,
          BITS: tl.constexpr, GS: tl.constexpr, XSTRIDE: tl.constexpr, OUT_STRIDE: tl.constexpr):
    """Item x 32 columns: rotated activation rows (one per pair) x the expert's down rows -> OUT."""

    it = tl.program_id(0)
    if it < tl.load(COUNTS):
        e = tl.load(ITEMS + it * 3)
        if e < EXPERTS:
            first = tl.load(ITEMS + it * 3 + 1)
            cnt = tl.load(ITEMS + it * 3 + 2)
            cols = tl.program_id(1) * 32 + tl.arange(0, 32)
            lanes = tl.arange(0, 16)
            pid = tl.load(MEMBERS + first + lanes, mask=lanes < cnt, other=-1)
            ok = (lanes < cnt) & (pid >= 0)
            acc = _affine_group(X, W, S, B, e, tl.maximum(pid, 0), ok, cols, N, K, BITS, GS, XSTRIDE)
            tl.store(OUT + pid[:, None] * OUT_STRIDE + cols[None, :],
                     acc.to(OUT.dtype.element_ty), mask=ok[:, None] & (cols[None, :] < N))


@dataclass
class MooneyExperts:
    """A layer's routed experts (E entries, 2-bit g128 rotated) plus its shared expert's Affs."""

    gate: tuple                        # (words int32 [E, N, KW], scales fp16 [E, N, NG], biases fp16)
    up: tuple
    down: tuple
    bits: int                          # the routed experts' shared MLX format
    group: int
    rot_gate: Rotation
    rot_up: Rotation
    rot_down: Rotation
    shared_gate: Aff
    shared_up: Aff
    shared_down: Aff
    experts: int
    width: int                         # routed intermediate width
    dims: int                          # expert input dim
    shared_width: int
    kernel: str = "mooney"

    def gate_up(self, xg: torch.Tensor, xu: torch.Tensor, plan, out: torch.Tensor) -> None:
        """Rotated inputs xg/xu [rows, D] -> out [pairs, width] bf16 for the plan's routed pairs."""

        wg, sg, bg = self.gate
        wu, su, bu = self.up
        grid = (plan.items.shape[0], triton.cdiv(self.width, 32))
        _gateup[grid](xg, xu, wg, sg, bg, wu, su, bu, plan.items, plan.counts, plan.members, out,
                      N=self.width, K=self.dims, EXPERTS=self.experts, BITS=self.bits,
                      GS=self.group, SLOTS=plan.slots, XSTRIDE=xg.stride(0), num_warps=4)

    def down_proj(self, act_rot: torch.Tensor, plan, out: torch.Tensor) -> None:
        """Rotated activations [pairs, width] -> out [pairs, dims] in ``out``'s dtype."""

        w, s, b = self.down
        grid = (plan.items.shape[0], triton.cdiv(self.dims, 32))
        _down[grid](act_rot, w, s, b, plan.items, plan.counts, plan.members, out,
                    N=self.dims, K=self.width, EXPERTS=self.experts, BITS=self.bits,
                    GS=self.group, XSTRIDE=act_rot.stride(0), OUT_STRIDE=out.stride(0), num_warps=4)

    def nbytes(self) -> int:
        return sum(int(t.numel()) * int(t.element_size()) for t in self.gate + self.up + self.down)
