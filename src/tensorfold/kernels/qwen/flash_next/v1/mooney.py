"""Mooney kernels for the Metal path: the manifest's segmented rotation (fp32, one bf16 rounding) and
the routed experts' matmuls taking per-projection inputs, beside the existing any-width affine helpers.

Same semantics as families/qwen4_exp/cuda/mooney.py: per contiguous block ``x_b -> H_b * (signs_b * x_b)``
with ``H_b`` the normalized Sylvester Walsh-Hadamard in fp32, rounded once to bf16. Gate and up rotate
the activation separately (each projection's own signs); down rotates each pair's activation. Scales and
biases are read at their stored dtype (fp16 in Mooney packs) into fp32.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.qwen.flash_next.v1.base import (LANE_CODES, QDOT_HEADER, count, kernel)


# ---------------------------------------------------------------------------
# segmented sign + normalized Hadamard butterfly; threadgroup (s, r): segment s of row r
# BS threads -> BS/2 threads each hold a pair per stage. fp32 arithmetic, one bf16 store.

_ROT = r"""
  // threadgroup (s, r): BS values of X[r, OFF:OFF+BS] -> OUT[r, OFF:OFF+BS] rotated in fp32
  const int r = int(threadgroup_position_in_grid.y);
  const int t = int(thread_position_in_threadgroup.x);           // 0 .. BS/2 - 1
  threadgroup float buf[BS];
  buf[t] = float(X[size_t(r) * K + OFF + t]) * float(SIGNS[OFF + t]);
  buf[t + BS / 2] = float(X[size_t(r) * K + OFF + t + BS / 2]) * float(SIGNS[OFF + t + BS / 2]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int h = 1; h < BS; h <<= 1) {
    const int i = (t / h) * (2 * h) + t % h;
    const float a = buf[i], b = buf[i + h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[i] = a + b;
    buf[i + h] = a - b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  OUT[size_t(r) * BS + t] = bfloat(buf[t] * INV_SQRT_NORM);
  OUT[size_t(r) * BS + t + BS / 2] = bfloat(buf[t + BS / 2] * INV_SQRT_NORM);
"""


def rotate(x: mx.array, signs: mx.array, segments: list[tuple[int, int]], *, dims: int) -> mx.array:
    """x [R, K] bf16 -> rotated bf16: each (offset, size) segment signed and Hadamard-transformed."""

    if x.ndim != 2:
        x = x.reshape(-1, x.shape[-1])
    rows = int(x.shape[0])
    signs = signs.astype(mx.float32)
    parts = []
    for off, size in segments:
        if size < 2 or size & (size - 1) or size > 1024:
            raise ValueError(f"rotation blocks must be powers of two up to 1024, got {size}")
        src = _ROT.replace("INV_SQRT_NORM", f"{size ** -0.5}f")
        run = kernel(f"mooney_rot_{size}", src, ["X", "SIGNS"], ["OUT"], header="")
        parts.append(run(inputs=[x, signs],
                         template=[("BS", size), ("K", dims), ("OFF", off)],
                         grid=(size // 2, rows, 1), threadgroup=(size // 2, 1, 1),
                         output_shapes=[(rows, size)], output_dtypes=[mx.bfloat16])[0])
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=-1)


# ---------------------------------------------------------------------------
# the affine expert inner loops, meta-typed (fp16 scales for Mooney, bf16 elsewhere)

def _meta(meta: str) -> str:
    return "half" if meta == "f16" else "bfloat"


_EXPERT_HELPERS = LANE_CODES + r"""
// gate/up rows for separate rotated inputs: gate against xg, up against xu (Mooney rotates per projection)
template <int BITS, int GSZ, int K, int RPS, typename MT>
inline void mooney_gateup_rows(const device uint32_t* GWp, const device uint32_t* UWp,
                               const device MT* GSp, const device MT* GBp,
                               const device MT* USp, const device MT* UBp, size_t first,
                               const device bfloat* xg, const device bfloat* xu,
                               uint lane, thread float* ag, thread float* au) {
  constexpr int VPT = lane_values(BITS), RB = K * BITS / 8, KG = K / GSZ;
  const device uint8_t* gw = (const device uint8_t*)GWp + first * RB;
  const device uint8_t* uw = (const device uint8_t*)UWp + first * RB;
  for (int v0 = int(lane) * VPT; v0 < K; v0 += 32 * VPT) {
    float xgv[VPT], xuv[VPT], sg = 0.0f, su = 0.0f;
    for (int i = 0; i < VPT; i++) { xgv[i] = float(xg[v0 + i]); sg += xgv[i]; }
    for (int i = 0; i < VPT; i++) { xuv[i] = float(xu[v0 + i]); su += xuv[i]; }
    for (int row = 0; row < RPS; row++) {
      float qg[VPT], qu[VPT];
      lane_codes<BITS, VPT>(gw + row * RB, v0, qg);
      lane_codes<BITS, VPT>(uw + row * RB, v0, qu);
      float dg = 0.0f, du = 0.0f;
      for (int i = 0; i < VPT; i++) { dg = fma(qg[i], xgv[i], dg); du = fma(qu[i], xuv[i], du); }
      const size_t at = (first + row) * KG + v0 / GSZ;
      ag[row] += fma(float(GSp[at]), dg, float(GBp[at]) * sg);
      au[row] += fma(float(USp[at]), du, float(UBp[at]) * su);
    }
  }
}
template <int BITS, int GSZ, int NI, typename MT>
inline void mooney_down_rows(const device uint32_t* W, const device MT* S, const device MT* B, size_t first,
                             const device bfloat* x, uint lane, thread float* out) {
  constexpr int VPT = lane_values(BITS), NC = (NI / VPT + 31) / 32, RB = NI * BITS / 8, KG = NI / GSZ;
  float xv[NC][VPT], sums[NC];
  for (int c = 0; c < NC; c++) {
    const int v0 = (c * 32 + int(lane)) * VPT;
    sums[c] = 0.0f;
    for (int i = 0; i < VPT; i++) { xv[c][i] = v0 < NI ? float(x[v0 + i]) : 0.0f; sums[c] += xv[c][i]; }
  }
  for (int row = 0; row < 8; row++) {
    const device uint8_t* w = (const device uint8_t*)W + (first + row) * RB;
    float acc = 0.0f;
    for (int c = 0; c < NC; c++) {
      const int v0 = (c * 32 + int(lane)) * VPT;
      if (v0 < NI) {
        float q[VPT];
        lane_codes<BITS, VPT>(w, v0, q);
        float d = 0.0f;
        for (int i = 0; i < VPT; i++) d = fma(q[i], xv[c][i], d);
        const size_t at = (first + row) * KG + v0 / GSZ;
        acc += fma(float(S[at]), d, float(B[at]) * sums[c]);
      }
    }
    out[row] = acc;
  }
}
"""

_MOONEY_GATEUP = r"""
  // expert_gateup's slots and picks; gate reads XG, up reads XU (the rotation's per-projection inputs)
  const uint lane = thread_index_in_simdgroup;
  const uint g = simdgroup_index_in_threadgroup;
  const int p = int(threadgroup_position_in_grid.z);
  constexpr int SLOTS = TOPK + SHARED;
  const int r = p / SLOTS, slot = p % SLOTS;
  const bool shared = slot == TOPK;
  float picked[TOPK];
  const size_t e = shared ? 0 : size_t(simd_topk<NE>(LOGITS + r * NL, slot, lane, picked));
  if (!shared && threadgroup_position_in_grid.y == 0 && g == 0 && lane == 0) {
    PICK[r * TOPK + slot] = uint32_t(e);
    if (slot == TOPK - 1) {
      float total = 0.0f;
      float ex[TOPK];
      for (int kk = 0; kk < TOPK; kk++) { ex[kk] = metal::exp(picked[kk] - picked[0]); total += ex[kk]; }
      for (int kk = 0; kk < TOPK; kk++) WTS[r * TOPK + kk] = float(bfloat(ex[kk] / total));
    }
  }
  const int row0 = int(threadgroup_position_in_grid.y) * (SG * RPS) + int(g) * RPS;
  float ag[RPS], au[RPS];
  for (int row = 0; row < RPS; row++) { ag[row] = 0.0f; au[row] = 0.0f; }
  if (shared) mooney_gateup_rows<SWB, SWG, K, RPS, SMT>(SGW, SUW, SGS, SGB, SUS, SUB, size_t(row0),
                                                      X + r * K, X + r * K, lane, ag, au);
  else mooney_gateup_rows<WB, WG, K, RPS, RMT>(GW, UW, GS, GB, US, UB, e * N + row0,
                                             XG + r * K, XU + r * K, lane, ag, au);
  for (int row = 0; row < RPS; row++) {
    const float gv = simd_sum(ag[row]), uv = simd_sum(au[row]);
    if (lane == 0) ACT[p * N + row0 + row] = bfloat(bsilu(float(bfloat(gv))) * float(bfloat(uv)));
  }
"""

_MOONEY_DOWN_Y = r"""
  // expert_down_y for any width: the routed pairs read ACTR (rotated activations), the shared slot ACT
  const uint lane = thread_index_in_simdgroup;
  const int R = rows[0];
  const int pair = int(threadgroup_position_in_grid.z) * SG + int(simdgroup_index_in_threadgroup);
  constexpr int SLOTS = TOPK + 1;
  if (pair >= R * SLOTS) return;
  const int r = pair / SLOTS, k = pair % SLOTS;
  const int d0 = int(threadgroup_position_in_grid.y) * 8;
  const bool shared = k == TOPK;
  const device bfloat* x = (shared ? ACT : ACTR) + (r * SLOTS + k) * NI;
  float out[8];
  if (shared) mooney_down_rows<SWB, SWG, NI, SMT>(SDW, SDS, SDB, size_t(d0), x, lane, out);
  else mooney_down_rows<WB, WG, NI, RMT>(DW, DS, DB, size_t(PICK[r * TOPK + k]) * D + d0, x, lane, out);
  for (int row = 0; row < 8; row++) {
    const float v = simd_sum(out[row]);
    if (lane == 0) Y[(r * SLOTS + k) * D + d0 + row] = bfloat(v);
  }
"""


def _fmt(linears: list) -> tuple[int, int, str]:
    """(bits, group, meta) shared by linears a kernel reads together; meta from the scales' dtype."""

    found = {(int(l.bits), int(l.group_size)) for l in linears}
    if len(found) != 1:
        raise ValueError(f"projections read together need one format, got {sorted(found)}")
    bits, group = found.pop()
    meta = "f16" if linears[0].scales.dtype == mx.float16 else "bf16"
    if any(l.scales.dtype != linears[0].scales.dtype for l in linears):
        raise ValueError("scales read together must share a dtype")
    return bits, group, meta


def mooney_gateup(x: mx.array, xg: mx.array, xu: mx.array, logits: mx.array, top_k: int, experts: int,
                  gate: Any, up: Any, shared: tuple[Any, Any] | None = None,
                  *, rows_per_simdgroup: int = 4, simdgroups: int = 2) -> tuple:
    """expert_gateup's shape of result with per-projection inputs: xg for gate, xu for up, x for the shared."""

    rows, dims = xg.shape
    width = int(gate.weight.shape[1])
    extra = 1 if shared is not None else 0
    sg, su = shared if shared is not None else (gate, up)
    wb, wg, rmt = _fmt([gate, up])
    swb, swg, smt = _fmt([sg, su])
    src = (_MOONEY_GATEUP.replace("RMT", _meta(rmt)).replace("SMT", _meta(smt)))
    run = kernel(f"mooney_gateup_{rmt}_{smt}", src,
                 ["X", "XG", "XU", "LOGITS", "GW", "GS", "GB", "UW", "US", "UB",
                  "SGW", "SGS", "SGB", "SUW", "SUS", "SUB"], ["ACT", "PICK", "WTS"],
                 header=QDOT_HEADER + _EXPERT_HELPERS)
    return tuple(run(inputs=[x, xg, xu, logits, gate.weight, gate.scales, gate.biases,
                             up.weight, up.scales, up.biases,
                             sg.weight, sg.scales, sg.biases, su.weight, su.scales, su.biases],
                     template=[("K", dims), ("N", width), ("TOPK", top_k), ("SHARED", extra),
                               ("NE", experts), ("NL", int(logits.shape[-1])),
                               ("RPS", rows_per_simdgroup), ("SG", simdgroups),
                               ("WB", wb), ("WG", wg), ("SWB", swb), ("SWG", swg)],
                     grid=(32 * simdgroups, width // (rows_per_simdgroup * simdgroups),
                           rows * (top_k + extra)),
                     threadgroup=(32 * simdgroups, 1, 1),
                     output_shapes=[(rows, top_k + extra, width), (rows, top_k), (rows, top_k)],
                     output_dtypes=[mx.bfloat16, mx.uint32, mx.float32]))


def mooney_down_y(act_rot: mx.array, act_plain: mx.array, picks: mx.array, down: Any, shared: Any,
                  *, simdgroups: int = 2) -> mx.array:
    """expert_down_y's result; routed pairs read the rotated ``act_rot``, the shared slot ``act_plain``."""

    rows, slots, width = act_rot.shape
    top_k = int(picks.shape[-1])
    dims = int(down.weight.shape[1])
    wb, wg, rmt = _fmt([down])
    swb, swg, smt = _fmt([shared])
    src = _MOONEY_DOWN_Y.replace("RMT", _meta(rmt)).replace("SMT", _meta(smt))
    run = kernel(f"mooney_down_{rmt}_{smt}", src,
                 ["ACTR", "ACT", "PICK", "DW", "DS", "DB", "SDW", "SDS", "SDB", "rows"], ["Y"],
                 header=QDOT_HEADER + _EXPERT_HELPERS)
    return run(inputs=[act_rot, act_plain, picks, down.weight, down.scales, down.biases,
                       shared.weight, shared.scales, shared.biases, count(rows)],
               template=[("NI", width), ("D", dims), ("TOPK", top_k), ("SG", simdgroups),
                         ("WB", wb), ("WG", wg), ("SWB", swb), ("SWG", swg)],
               grid=(32 * simdgroups, dims // 8, -(-rows * slots // simdgroups)),
               threadgroup=(32 * simdgroups, 1, 1),
               output_shapes=[(rows, slots, dims)], output_dtypes=[mx.bfloat16])[0]
