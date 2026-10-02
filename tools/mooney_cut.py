"""Cut a Mooney pack to N layers for bring-up tests: a new, loadable pack (never mutates the source).

Writes <out>/config.json (pruned layer_types/ple_layer_ids/quantization keys), the rotation manifest
pruned to the kept layers, a single model.safetensors rebuilt by streaming each kept tensor's raw
payload bytes (dtypes preserved verbatim, bounded 64 MiB chunks — no decode, no bf16 reinterpretation),
and model.safetensors.index.json over it. n-gram (PLE) shards of kept layers are kept whole: row ids
are hash % prime + head_offset over the full table, so a partial table would corrupt lookups.
Auxiliary files (tokenizer, tensor_map.json, …) are symlinked.

Usage:
    python tools/mooney_cut.py <mooney_pack_dir> <out_dir> [--layers 4]

The destination directory must not already exist.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

CHUNK = 64 * 1024 * 1024


def _read_shard_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def main() -> int:
    ap = argparse.ArgumentParser(description="cut a Mooney MLX pack to --layers layers")
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()

    src, dst = args.src.resolve(), args.dst
    index_path = src / "model.safetensors.index.json"
    if not index_path.is_file() or not (src / "mooney_rotation.json").is_file():
        sys.exit(f"{src}: not a Mooney pack (need model.safetensors.index.json + mooney_rotation.json)")
    if dst.exists():
        sys.exit(f"{dst}: exists; refusing to clobber (remove it or pick a new dir)")
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    cfg = json.loads((src / "config.json").read_text())
    tc = cfg["text_config"]
    layers = int(tc["num_hidden_layers"])
    keep = min(args.layers, layers)
    if keep >= layers:
        sys.exit(f"--layers {keep} >= the pack's {layers}; nothing to cut")

    mtp_layers = int(tc.get("mtp_num_hidden_layers") or 1)

    def kept(name: str) -> bool:
        for pre, n in (("language_model.model.layers.", keep), ("language_model.mtp.layers.", mtp_layers)):
            if name.startswith(pre):
                return int(name[len(pre):].split(".", 1)[0]) < n
        return True                     # embed, head, norms, vision, globals, mtp non-layer weights

    kept_names = [n for n in weight_map if kept(n)]
    print(f"keeping {len(kept_names)} of {len(weight_map)} tensors ({keep} of {layers} layers)")

    # --- config: prune layer types, ple ids, and the quantization map's dropped entries ----------
    tc["num_hidden_layers"] = keep
    tc["layer_types"] = [k for i, k in enumerate(tc["layer_types"]) if i < keep]
    tc["ple_layer_ids"] = [i for i in tc.get("ple_layer_ids") or [] if i <= keep]
    quant = cfg.get("quantization") or {}
    cfg["quantization"] = {k: v for k, v in quant.items()
                           if not (isinstance(k, str) and k.startswith("layers.")
                                   and int(k[len("layers."):].split(".", 1)[0]) >= keep)}

    # --- manifest: only the kept layers' entries -------------------------------------------------
    man = json.loads((src / "mooney_rotation.json").read_text())
    man["weights"] = {n: e for n, e in man["weights"].items()
                      if int(n.split(".layers.")[1].split(".")[0]) < keep}
    man.pop("inverses", None)

    # --- safetensors: stream each kept tensor's bytes into one shard, dtype/shape verbatim -------
    headers: dict[str, dict] = {}
    for i, name in enumerate(sorted(kept_names)):
        shard_path = src / weight_map[name]
        key = str(shard_path)
        if key not in headers:
            headers[key] = _read_shard_header(shard_path)
        hdr, _ = headers[key]
        if name not in hdr:
            sys.exit(f"{name}: not in {shard_path.name}'s header")
    dst.mkdir(parents=True)

    # first pass: new header with re-based offsets
    out_entries: dict[str, dict] = {}
    at = 0
    order = sorted(kept_names)
    for name in order:
        entry = headers[str(src / weight_map[name])][0][name]
        size = entry["data_offsets"][1] - entry["data_offsets"][0]
        out_entries[name] = {"dtype": entry["dtype"], "shape": entry["shape"],
                             "data_offsets": [at, at + size]}
        at += size
        at = (at + 7) & ~7                     # safetensors: every tensor's start is 8-aligned
    out_header = json.dumps(out_entries, separators=(",", ":")).encode()
    pad = -len(out_header) % 8
    out_header += b" " * pad

    out_path = dst / "model.safetensors"
    total = 0
    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(out_header)))
        out.write(out_header)
        for i, name in enumerate(order):
            entry = headers[str(src / weight_map[name])][0][name]
            payload_off = headers[str(src / weight_map[name])][1]
            start, end = entry["data_offsets"]
            with open(src / weight_map[name], "rb") as f:
                f.seek(payload_off + start)
                left = end - start
                while left:
                    chunk = f.read(min(CHUNK, left))
                    if not chunk:
                        sys.exit(f"{name}: short read in {weight_map[name]}")
                    out.write(chunk)
                    left -= len(chunk)
                    total += len(chunk)
            pad = (-(end - start)) & 7
            if pad:
                out.write(b"\0" * pad)
            if i % 200 == 0:
                print(f"  {i}/{len(order)} {name} ({total / 2**30:.1f} GiB)", flush=True)
    idx_out = {"metadata": {"total_size": total},
               "weight_map": {n: "model.safetensors" for n in order}}
    (dst / "model.safetensors.index.json").write_text(json.dumps(idx_out, indent=2))
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    (dst / "mooney_rotation.json").write_text(json.dumps(man, indent=2))
    for f in src.iterdir():
        if f.name not in {"model.safetensors.index.json", "config.json", "mooney_rotation.json"} \
                and not f.name.startswith("model-") and f.suffix != ".safetensors" and f.is_file():
            dst_f = dst / f.name
            if not dst_f.exists():
                dst_f.symlink_to(f)
    print(f"wrote {dst}: {total / 2**30:.1f} GiB, {len(order)} tensors, {keep} layers, "
          f"ple ids {tc['ple_layer_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
