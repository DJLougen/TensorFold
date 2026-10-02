"""The Mooney checkpoint contract: `mooney_rotation.json` beside the weights, fail-closed.

Mooney packs (FORMAT.md v1) keep the routed experts as rotated ternary codes: MLX affine 2-bit in
groups of 128 in a rotated basis, while the dense projections, the n-gram table and the MTP head are
affine 8-bit in groups of 32 and small tensors stay unquantized. Every quantized tensor's scales and
biases are fp16.

Before a rotated expert matmul, the activation's input dim goes through the manifest's segmented
rotation: for each contiguous block, x_b -> H_b * (signs_b * x_b) in fp32 with one rounding to bf16,
where H_b is the normalized Sylvester Walsh-Hadamard (H[r][c] = (-1)^popcount(r & c) / sqrt(B)).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NAME = "mooney_rotation.json"
FORMAT = "lowbitflash.rot"
VERSION = 1
TRANSFORM_ORDER = "signs_first_then_hadamard"
HADAMARD_KIND = "sylvester_normalized"
# rotated weights must be inside a layer's switch_mlp expert stack (the only verified rotated path)
_ROTATED = re.compile(r"^(?:language_model\.)?(?:model\.)?layers\.\d+\.mlp\.switch_mlp\.(?:gate_proj|up_proj|down_proj)$|"
                      r"^(?:language_model\.)?mtp\.layers\.\d+\.mlp\.switch_mlp\.(?:gate_proj|up_proj|down_proj)$")
MAX_DIM = 1 << 22          # a rotated input dim larger than this is a malformed manifest, not a checkpoint


@dataclass(frozen=True)
class RotSpec:
    """One rotated tensor's segmented transform: contiguous blocks of ``blocks`` sizes, signs length sum(blocks)."""

    name: str                     # the MLX checkpoint name (the .weight suffix implied)
    blocks: tuple[int, ...]       # contiguous power-of-two block sizes partitioning the input dim
    signs: tuple[int, ...]        # +/-1, one per input element, in block order
    gguf: str = ""                # the GGUF tensor the pack folded (diagnostics only)

    @property
    def dim(self) -> int:
        return sum(self.blocks)

    def segments(self) -> list[tuple[int, int]]:
        """(offset, size) of each block in input order."""

        out, at = [], 0
        for size in self.blocks:
            out.append((at, size))
            at += size
        return out


@dataclass(frozen=True)
class Manifest:
    """A parsed mooney_rotation.json that passed every structural check."""

    path: Path
    weights: tuple[RotSpec, ...]
    inverses: tuple[RotSpec, ...]

    def for_weight(self, name: str) -> RotSpec | None:
        """The rotation for an MLX weight path (with or without its .weight suffix), or None."""

        wanted = name[:-len(".weight")] if name.endswith(".weight") else name
        for spec in self.weights:
            if spec.name == wanted:
                return spec
        return None

    def fingerprint(self) -> str:
        """A short digest of the transform content (snapshot keys)."""

        h = hashlib.sha256(self.path.read_bytes())
        return h.hexdigest()[:12]


def is_mooney(model_dir: str | Path) -> bool:
    """Whether the checkpoint directory carries the Mooney rotation manifest."""

    return (Path(model_dir) / NAME).is_file()


def _ints(value: Any, what: str) -> list[int]:
    if not isinstance(value, list) or any(type(v) is not int for v in value):
        raise ValueError(f"mooney_rotation.json: {what} must be a list of integers")
    return value


def _spec(name: str, entry: dict[str, Any], model_dir: Path, index_names: set[str] | None) -> RotSpec:
    """One manifest weight entry -> RotSpec, refusing anything outside the version-1 contract."""

    if not isinstance(entry, dict):
        raise ValueError(f"mooney_rotation.json: weights[{name!r}] must be an object")
    unknown = set(entry) - {"blocks", "signs", "signs_file", "signs_offset", "signs_length", "gguf_name"}
    if unknown:
        raise ValueError(f"mooney_rotation.json: weights[{name!r}] has unknown field(s): {sorted(unknown)}")
    if not isinstance(name, str) or not _ROTATED.match(name):
        raise ValueError(f"mooney_rotation.json: {name!r} is not a routed-expert projection this reader rotates")
    if index_names is not None and name + ".weight" not in index_names:
        raise ValueError(f"mooney_rotation.json: rotated weight {name!r} is not in the checkpoint's index")
    blocks = _ints(entry.get("blocks"), f"weights[{name!r}].blocks")
    if not blocks:
        raise ValueError(f"mooney_rotation.json: {name!r} has no blocks")
    for b in blocks:
        # the reference format admits pow2 blocks that are multiples of 128 (128..8192)
        if b < 128 or b > 8192 or b & (b - 1) or b % 128:
            raise ValueError(f"mooney_rotation.json: {name!r} block size {b} is not a power-of-two multiple of 128")
    dim = sum(blocks)
    if dim > MAX_DIM:
        raise ValueError(f"mooney_rotation.json: {name!r} blocks sum to {dim} inputs (implausible)")
    inline = "signs" in entry
    external = "signs_file" in entry
    if inline and external:
        raise ValueError(f"mooney_rotation.json: {name!r} carries both inline signs and a signs_file")
    if not inline and not external:
        raise ValueError(f"mooney_rotation.json: {name!r} has no signs")
    if inline:
        signs = _ints(entry["signs"], f"weights[{name!r}].signs")
        if len(signs) != dim:
            raise ValueError(f"mooney_rotation.json: {name!r} has {len(signs)} signs for {dim} inputs")
    else:
        file_name = entry["signs_file"]
        if not isinstance(file_name, str) or Path(file_name).name != file_name:
            raise ValueError(f"mooney_rotation.json: {name!r} signs_file {file_name!r} is not a plain file name")
        offset = entry.get("signs_offset", 0)
        length = entry.get("signs_length")
        if type(offset) is not int or type(length) is not int or offset < 0 or length != dim:
            raise ValueError(f"mooney_rotation.json: {name!r} signs_offset/signs_length do not cover {dim} inputs")
        blob = model_dir / file_name
        if not blob.is_file() or blob.stat().st_size < offset + length:
            raise ValueError(f"mooney_rotation.json: {name!r} signs beyond {file_name}")
        with open(blob, "rb") as f:
            f.seek(offset)
            signs = list(struct_int8(f.read(length), name))
    if any(s not in (-1, 1) for s in signs):
        raise ValueError(f"mooney_rotation.json: {name!r} signs must be +1 or -1")
    gguf = entry.get("gguf_name", "")
    if gguf and not isinstance(gguf, str):
        raise ValueError(f"mooney_rotation.json: {name!r} gguf_name is not a string")
    return RotSpec(name, tuple(blocks), tuple(int(s) for s in signs), gguf)


def struct_int8(raw: bytes, name: str) -> list[int]:
    """An i8 signs blob -> ints (-1/1 only); anything else fails closed."""

    out = []
    for byte in raw:
        v = byte - 256 if byte > 127 else byte
        if v not in (-1, 1):
            raise ValueError(f"mooney_rotation.json: {name!r} signs_file byte is not +1 or -1")
        out.append(v)
    return out


def manifest(model_dir: str | Path, index_names: set[str] | None = None) -> Manifest:
    """Read and fully validate ``mooney_rotation.json``; every structural defect is a ValueError."""

    path = Path(model_dir) / NAME
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"{path.name}: unreadable manifest ({e})") from e
    if not isinstance(doc, dict):
        raise ValueError("mooney_rotation.json: the manifest must be a JSON object")
    if doc.get("format") != FORMAT:
        raise ValueError(f"mooney_rotation.json: format {doc.get('format')!r} is not {FORMAT!r}")
    if doc.get("version") != VERSION:
        raise ValueError(f"mooney_rotation.json: version {doc.get('version')!r} is not supported (v{VERSION} only)")
    hadamard = doc.get("hadamard")
    if not isinstance(hadamard, dict) or hadamard.get("kind") != HADAMARD_KIND:
        raise ValueError(f"mooney_rotation.json: hadamard.kind must be {HADAMARD_KIND!r}")
    transform = doc.get("transform")
    if not isinstance(transform, dict) or transform.get("order") != TRANSFORM_ORDER:
        raise ValueError(f"mooney_rotation.json: transform.order must be {TRANSFORM_ORDER!r}")
    weights = doc.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("mooney_rotation.json: no weights map")
    inverses = doc.get("inverses") or {}
    if not isinstance(inverses, dict):
        raise ValueError("mooney_rotation.json: inverses must be an object")
    parsed = tuple(_spec(name, entry, Path(model_dir), index_names) for name, entry in weights.items())
    inv = tuple(_spec(name, entry, Path(model_dir), index_names) for name, entry in inverses.items())
    if inv:
        raise ValueError("mooney_rotation.json: inverse (latent-table) rotations are not read here; "
                         "Mooney v1 ships none")
    return Manifest(path, parsed, inv)


def refusal(model_dir: str | Path, index_names: set[str] | None = None) -> str | None:
    """The reason this directory's manifest is outside the contract, or None when it parses."""

    try:
        manifest(model_dir, index_names)
    except ValueError as e:
        return str(e)
    return None


def module_spec(config: dict[str, Any], path: str) -> tuple[int, int] | None:
    """(bits, group) the module at ``path`` stores, or None when unquantized.

    MLX's per-module semantics (exact canonical keys, else the checkpoint's default) plus ``modules``
    glob entries a Mooney pack may carry for the shared expert stack.
    """

    import fnmatch

    from tensorfold.quantization import canonical_path, quantization_block, resolve_affine

    block = quantization_block(config)
    if block is None:
        return None
    wanted = canonical_path(path)
    value: Any = ...
    for key, entry in block.items():
        if key == "modules" or canonical_path(str(key)) != wanted:
            continue
        value = entry
        break
    if value is ...:
        modules = block.get("modules")
        if isinstance(modules, dict):
            for pattern, entry in modules.items():
                if isinstance(pattern, str) and fnmatch.fnmatchcase(wanted, canonical_path(pattern)):
                    value = entry
                    break
    if value is ...:
        spec = resolve_affine(config, path)
        return None if spec is None else (spec.bits, spec.group_size)
    if value is False or value == {}:
        return None
    if value is True:
        spec = resolve_affine(config)
        return None if spec is None else (spec.bits, spec.group_size)
    if isinstance(value, dict):
        bits = value.get("bits")
        if bits is None:
            return None
        return int(bits), int(value.get("group_size", 64))
    raise ValueError(f"invalid per-module quantization entry for {path}: {value!r}")
