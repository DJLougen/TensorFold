"""Load a Mooney MLX-format pack on one GPU: rotated 2-bit group-128 experts, affine 8-bit
group-32 dense tensors, fp16 scales and biases, unquantized norm/gate/scalar tensors.

The tensors keep the checkpoint's bytes: no repacking, no rounding. The rotation metadata is the
checkpoint's own ``mooney_rotation.json`` (families.qwen4_exp.mooney), validated before weights load
and checked again against each tensor's shape here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .. import mooney as mf
from ..host_table import open_table, shard_keys
from .. import read_config
from .reader import _Reader, norms_around_one
from .weight_types import (AttnW, Config, GDNW, HC, LayerW, MoEW, MTPW, PLEW, Weights,
                           draft_token_ids)


def dequant(name: str, words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, bits: int,
            group: int) -> torch.Tensor:
    """MLX affine triple -> fp32 values s * q + b (exact for fp16/fp32 scalars)."""

    k_words = words.shape[-1]
    if words.dtype != torch.int32 or k_words * 32 % bits:
        raise ValueError(f"{name}: packed words do not match {bits} bits")
    k = k_words * 32 // bits
    if scales.shape[-1] * group != k:
        raise ValueError(f"{name}: {scales.shape[-1]} groups of {group} do not span {k} inputs")
    w = words.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(32 // bits, device=words.device, dtype=torch.int64) * bits
    q = ((w[..., None] >> shifts) & ((1 << bits) - 1)).reshape(*words.shape[:-1], k).to(torch.float32)
    s = scales.to(torch.float32).repeat_interleave(group, dim=-1)
    b = biases.to(torch.float32).repeat_interleave(group, dim=-1)
    return q * s + b


def load(model_dir: Path, device: str = "cuda", *, mtp: bool = True,
         draft_vocab: int | str | None = None, ple_on_ssd: bool = False,
         table_reads: list | None = None) -> Weights:
    """The Mooney pack's weights on ``device``; tp is refused by the caller before this runs."""

    import time

    from . import mooney as ck

    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    cfg_json = read_config(model_dir)
    rd = _Reader(model_dir, device)
    prefix = "language_model." if rd.has("language_model.model.embed_tokens.weight") else ""
    man = mf.manifest(model_dir, set(rd.where))          # every rotated name checked against the index
    rotated = {spec.name[len(prefix):] if spec.name.startswith(prefix) else spec.name: spec
               for spec in man.weights}
    around_one = norms_around_one(rd, prefix + "model.", list(range(cfg.layers)))

    def raw(name: str) -> torch.Tensor:
        return rd.get(prefix + name)

    def spec(name: str) -> tuple[int, int] | None:
        """(bits, group) the module that owns ``name``'s weight stores, None when unquantized."""

        return mf.module_spec(cfg_json, name)

    def triple(name: str, want: tuple[int, int] | None = None) -> tuple:
        w = raw(name + ".weight")
        w = w.view(torch.int32) if w.dtype != torch.int32 else w
        s, b = raw(name + ".scales"), raw(name + ".biases")
        got = spec(name)
        if want is not None and got != want:
            raise ValueError(f"{name}: Mooney expects {want[0]}-bit groups of {want[1]}, "
                             f"the pack's config resolves {got}")
        if s.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"{name}: scales must be fp16/bf16/fp32, not {s.dtype}")
        if b.dtype != s.dtype:
            raise ValueError(f"{name}: biases' dtype {b.dtype} differs from scales' {s.dtype}")
        return w, s, b

    def aff(name: str, want: tuple[int, int] | None = None) -> ck.Aff:
        """One checkpoint linear: an affine triple at its stored format, or a dense weight."""

        if rd.has(prefix + name + ".scales"):
            w, s, b = triple(name, want)
            fmt = spec(name)
            if fmt is None:
                raise ValueError(f"{name}: scales without a quantization spec")
            return ck.Aff(w, s, b, fmt[0], fmt[1], "mlx")
        if want is not None:
            raise ValueError(f"{name}: expected {want[0]}-bit weights, found an unquantized tensor")
        w = raw(name + ".weight")
        if w.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"{name}: {w.dtype} without scales is not a weight Mooney reads")
        return ck.Aff(w.contiguous(), None, None, 0, 0, "dense")

    def stack(*names: str):
        return ck.stack_of([aff(n) for n in names])

    def cscale(name: str) -> torch.Tensor:
        w = raw(name).float()
        return (w if around_one else 1.0 + w).contiguous()

    def table_scale(base: str, field: str) -> float:
        name = base + "ngram_embedding." + field
        if not rd.has(prefix + name):
            return 1.0
        value = raw(name)
        if value.numel() != 1:
            raise ValueError(f"{name}: expected one n-gram table scale")
        return float(value.float().reshape(-1)[0])

    def hc(name: str, inject: bool) -> HC:
        """input_mix_weight_down (+ block_inject_weight) then input_mix_weight_up, affine 8-bit."""

        parts = [aff(name + ".input_mix_weight_down", (8, 32))]
        if inject:
            parts.append(aff(name + ".block_inject_weight", (8, 32)))
        down = ck.stack_of(parts)
        up = aff(name + ".input_mix_weight_up", (8, 32))
        return HC(down, up, cscale(name + ".hc_norm.weight"), inject, down, up)

    def gdn(name: str) -> GDNW:
        proj = stack(name + ".in_proj_qkv", name + ".in_proj_z", name + ".in_proj_b",
                     name + ".in_proj_a")
        conv = raw(name + ".conv1d.weight").reshape(cfg.conv_dim, cfg.conv_kernel).to(torch.bfloat16)
        return GDNW(proj, conv.contiguous(), raw(name + ".A_log").float().contiguous(),
                    raw(name + ".dt_bias").float().contiguous(),
                    raw(name + ".norm.weight").to(torch.bfloat16).contiguous(),
                    aff(name + ".out_proj"))

    def attention(name: str) -> AttnW:
        proj = stack(name + ".q_proj", name + ".k_proj", name + ".v_proj",
                     name + ".indexer.index_qk_proj")
        return AttnW(proj, cscale(name + ".q_norm.weight"), cscale(name + ".k_norm.weight"),
                     cscale(name + ".indexer.q_layernorm.weight"),
                     cscale(name + ".indexer.k_layernorm.weight"), aff(name + ".o_proj"))

    def rotation(name: str, words: torch.Tensor, scales: torch.Tensor) -> ck.Rotation | None:
        """The rotation the manifest declares for ``name``, checked against the weight's input dim."""

        sp = rotated.get(name)
        if sp is None:
            return None
        fmt = spec(name)
        if fmt is None:
            raise ValueError(f"{name}: a rotated weight must be quantized")
        k = words.shape[-1] * 32 // fmt[0]
        if scales.shape[-1] * fmt[1] != k:
            raise ValueError(f"{name}: {fmt[0]}-bit groups of {fmt[1]} do not span {k} inputs")
        if sp.dim != k:
            raise ValueError(f"{name}: the manifest's rotation spans {sp.dim} inputs, the weight has {k}")
        return ck.Rotation.build(sp, device)

    def moe(name: str) -> MoEW:
        """Routed experts (rotated 2-bit g128 when the manifest covers them) plus the shared expert."""

        gate_rows = raw(name + ".gate.weight").to(torch.bfloat16)
        sw, ss, sb = triple(name + ".shared_expert_gate")
        sfmt = spec(name + ".shared_expert_gate") or (8, 32)
        shared_gate = dequant(name + ".shared_expert_gate", sw, ss, sb, *sfmt).to(torch.bfloat16)
        router = torch.cat([gate_rows, shared_gate]).contiguous()
        gate, up, down = (triple(name + ".switch_mlp." + p)
                          for p in ("gate_proj", "up_proj", "down_proj"))
        efmt = spec(name + ".switch_mlp.gate_proj")
        fmts = {spec(name + ".switch_mlp." + p) for p in ("gate_proj", "up_proj", "down_proj")}
        if len(fmts) != 1 or efmt is None:
            raise ValueError(f"{name}: the routed experts' formats differ ({fmts}); Mooney stores them "
                             "uniformly")
        rg = rotation(name + ".switch_mlp.gate_proj", *gate[0:2])
        ru = rotation(name + ".switch_mlp.up_proj", *up[0:2])
        rdwn = rotation(name + ".switch_mlp.down_proj", *down[0:2])
        have = (rg, ru, rdwn)
        if any(r is not None for r in have):
            if not all(r is not None for r in have):
                raise ValueError(f"{name}: the manifest rotates only some of gate/up/down; all three or none")
            if efmt != (2, 128):
                raise ValueError(f"{name}: rotated experts must be 2-bit groups of 128, resolved {efmt}")
        elif efmt not in ((2, 128), (8, 32)):
            raise ValueError(f"{name}: unrotated routed experts must be 2/128 or 8/32, resolved {efmt}")
        shared = [aff(name + ".shared_expert." + p) for p in ("gate_proj", "up_proj", "down_proj")]
        experts = ck.MooneyExperts(gate, up, down, efmt[0], efmt[1], rg, ru, rdwn,
                                   shared[0], shared[1], shared[2],
                                   cfg.experts, cfg.moe_width, cfg.hidden, cfg.shared_width)
        return MoEW(router, experts)

    def ple_layer(name: str, ple_index: int) -> PLEW:
        ngram = cfg.ngram(ple_index)
        base = name + ".ple_embedding."
        ngram.check(raw(base + "layer_multipliers").cpu().numpy(), raw(base + "ngram_heads_offsets").cpu().numpy(),
                    raw(base + "ngram_heads_vocab_sizes").cpu().numpy())
        keys = shard_keys(prefix + base + "ngram_embedding", cfg.ngram_shards, rd.where)
        table = open_table(model_dir, [(rd.where[k + ".weight"], k) for k in keys],
                           lambda n: table_scale(base, n), ssd=ple_on_ssd)
        if table.rows != ngram.rows:
            raise ValueError(f"n-gram tables hold {table.rows} rows, expected {ngram.rows}")
        if getattr(table, "width", ngram.dims) != ngram.dims:
            raise ValueError(f"the n-gram rows hold {table.width} values, expected {ngram.dims}")
        if table_reads is not None and not ple_on_ssd:
            from tensorfold.cuda.direct_read import in_background

            in_background(table.prefetch, table_reads)
        conv = raw(name + ".conv1d.weight").reshape(cfg.streams * cfg.hidden, cfg.ple_kernel)
        return PLEW(table, aff(name + ".key_proj"), aff(name + ".value_proj"),
                    cscale(name + ".norm_key.weight"), cscale(name + ".norm_query.weight"),
                    cscale(name + ".norm_conv.weight"), conv.to(torch.bfloat16).contiguous(), ngram)

    def layer(i: int, base: str, kind: str, with_ple: bool) -> LayerW:
        linear = kind == "linear"
        entry = LayerW(i, linear, hc(base + ".attn_hyper_connection", True),
                       hc(base + ".mlp_hyper_connection", True),
                       gdn(base + ".linear_attn") if linear else None,
                       None if linear else attention(base + ".self_attn"), moe(base + ".mlp"))
        if with_ple and i in cfg.ple_layers:
            entry.ple = ple_layer(base + ".ple", cfg.ple_layers.index(i))
        return entry

    t0 = time.time()
    try:
        embed = aff("model.embed_tokens", (8, 32))
        loaded = []
        ahead = rd.layer_names(prefix, "model.", list(range(cfg.layers)), mtp)
        layer_events: list = []
        for k, i in enumerate(range(cfg.layers)):
            if len(layer_events) >= 2:
                layer_events.pop(0).synchronize()
            for names in ahead[k:k + 2]:
                rd.queue(names)
            loaded.append(layer(i, f"model.layers.{i}", cfg.layer_types[i], True))
            layer_events.append(torch.cuda.current_stream().record_event())
            rd.drop(ahead[k])
            rd.release()
            if i % 8 == 7:
                torch.cuda.empty_cache()
        mixer = hc("model.hyper_connection_mixer", False)
        head = aff("lm_head", (8, 32))
        draft_head, draft_ids = None, None
        ids = draft_token_ids(draft_vocab)
        if ids is not None:
            import numpy as np

            ids = ids[ids < cfg.vocab]
            ids = torch.from_numpy(np.asarray(ids, dtype=np.int64)).to(device)
            draft_ids = ids
            words, scales, biases = triple("lm_head")
            rows = dequant("lm_head", words.index_select(0, ids), scales.index_select(0, ids),
                           biases.index_select(0, ids), 8, 32).to(torch.bfloat16)
            from .bf16 import quantize4

            draft_head = quantize4(rows)
        inv = torch.tensor(cfg.rope_theta, dtype=torch.float64) ** (
            -torch.arange(0, cfg.rotary_dim // 2, dtype=torch.float64) / (cfg.rotary_dim // 2))
        w = Weights(cfg, embed, loaded, mixer, head, inv.to(torch.float32).to(device),
                    around_one=around_one)
        w.meta.update(rank=0, world=1, vocab_offset=0, full=cfg, mooney=man.fingerprint())
        w.mooney = man
        w.draft_head, w.draft_ids = draft_head, draft_ids
        if mtp and rd.has(prefix + "mtp.fc_embedding.weight"):
            w.mtp = MTPW(cscale("mtp.pre_fc_norm_embedding.weight"), cscale("mtp.pre_fc_norm_hidden.weight"),
                         aff("mtp.fc_embedding"), aff("mtp.fc_hidden"),
                         layer(-1, "mtp.layers.0", "attention", False),
                         hc("mtp.hyper_connection_mixer", False))
    except BaseException:
        rd.close()
        raise
    rd.close()
    rd.release()
    torch.cuda.empty_cache()
    w.meta["load_seconds"] = time.time() - t0
    return w
