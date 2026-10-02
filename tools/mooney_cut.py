"""Cut a Mooney pack to N layers for bring-up tests: a new, loadable pack (never mutates the source).

Writes <out>/config.json (pruned layer_types/ple_layer_ids/quantization keys), the rotation manifest
pruned to the kept layers, model.safetensors.index.json over a single rewritten model.safetensors,
and symlinks the tokenizer/auxiliary files. n-gram (PLE) shards of kept layers are kept whole — the
truncated model exercises the real fp16-scale table path; tables of dropped layers are dropped.

Usage:
    python tools/mooney_cut.py <mooney_pack_dir> <out_dir> [--layers 4]

The source pack is opened read-only via safetensors lazy loading; only kept tensors' bytes are read.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


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
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    cfg = json.loads((src / "config.json").read_text())
    tc = cfg["text_config"]
    layers = int(tc["num_hidden_layers"])
    keep = min(args.layers, layers)
    if keep >= layers:
        sys.exit(f"--layers {keep} >= the pack's {layers}; nothing to cut")

    def kept(name: str) -> bool:
        for pre, keep_n in (("language_model.model.layers.", keep), ("language_model.mtp.layers.",
                                                                   int(tc.get("mtp_num_hidden_layers") or 1))):
            if name.startswith(pre):
                return int(name[len(pre):].split(".", 1)[0]) < keep_n
        return True                     # embed, head, norms, vision, globals, mtp non-layer weights

    kept_names = [n for n in weight_map if kept(n)]
    print(f"keeping {len(kept_names)} of {len(weight_map)} tensors ({keep} of {layers} layers)")

    # --- config: prune layer Types, ple ids, and the quantization map's dropped entries ----------
    tc["num_hidden_layers"] = keep
    tc["layer_types"] = [k for i, k in enumerate(tc["layer_types"]) if i < keep]
    tc["ple_layer_ids"] = [i for i in tc.get("ple_layer_ids") or [] if i <= keep]
    quant = cfg.get("quantization") or {}
    pattern = "layers."
    cfg["quantization"] = {k: v for k, v in quant.items()
                           if not (isinstance(k, str) and k.startswith(pattern)
                                   and int(k[len(pattern):].split(".", 1)[0]) >= keep)}
    cfg["mooney_format_version"] = 1

    # --- manifest: only the kept layers' entries ------------------------------------------------
    man = json.loads((src / "mooney_rotation.json").read_text())
    man["weights"] = {n: e for n, e in man["weights"].items()
                      if int(n.split(".layers.")[1].split(".")[0]) < keep}
    man.pop("inverses", None)

    # --- safetensors: rewrite kept tensors into one shard ---------------------------------------
    dst.mkdir(parents=True, exist_ok=True)
    from safetensors import safe_open
    from safetensors.numpy import save_file

    arrays: dict[str, "object"] = {}
    import numpy as np
    handles: dict[str, object] = {}

    def tensor(name: str):
        path = weight_map[name]
        if path not in handles:
            handles[path] = safe_open(str(src / path), framework="numpy")
        return handles[path].get_tensor(name)

    for i, name in enumerate(sorted(kept_names)):
        arrays[name] = tensor(name)
        if i % 400 == 0:
            print(f"  {i}/{len(kept_names)} {name}", flush=True)
    save_file(arrays, str(dst / "model.safetensors"),
              metadata={"format": "pt"})
    idx_out = {"metadata": {"total_size": sum(a.nbytes for a in arrays.values())},
               "weight_map": {n: "model.safetensors" for n in arrays}}
    (dst / "model.safetensors.index.json").write_text(json.dumps(idx_out, indent=2))
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    (dst / "mooney_rotation.json").write_text(json.dumps(man, indent=2))
    for f in src.iterdir():
        if f.name not in {"model.safetensors.index.json", "config.json", "mooney_rotation.json"} \
                and not f.name.startswith("model-") and f.suffix != ".safetensors" and f.is_file():
            dst_f = dst / f.name
            if not dst_f.exists():
                dst_f.symlink_to(f)
    print(f"wrote {dst}: {sum(a.nbytes for a in arrays.values()) / 2**30:.1f} GiB, "
          f"{len(arrays)} tensors, {keep} layers, ple ids {tc['ple_layer_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
