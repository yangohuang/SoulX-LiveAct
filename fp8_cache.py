"""Offline safetensors cache for an already-quantized FP8 LiveAct DiT.

The cache contains no BF16 linear weight. It is loaded into a meta-device
model so startup never allocates or reads the original BF16 DiT shards.
"""

import json
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from fp8_gemm import FP8Linear, enable_fp8_gemm


FORMAT_VERSION = 1
FP8_PREFIX = "__fp8__."


def _fingerprint(source_paths):
    return [
        {"path": str(Path(path).resolve()), "size": Path(path).stat().st_size,
         "mtime_ns": Path(path).stat().st_mtime_ns}
        for path in source_paths
    ]


def _state_for_saving(module, *, exclude_blocks=False):
    tensors = {name: tensor.detach().cpu().contiguous()
               for name, tensor in module.state_dict().items()
               if isinstance(tensor, torch.Tensor) and not (exclude_blocks and name.startswith("blocks."))}
    for name, child in module.named_modules():
        if not isinstance(child, FP8Linear) or (exclude_blocks and name.startswith("blocks.")):
            continue
        if child._fp8_weight is None or child._fp8_weight_scale is None:
            raise ValueError(f"FP8 weight is not materialized: {name}")
        tensors[f"{FP8_PREFIX}{name}.weight_nk"] = child._fp8_weight.t().detach().cpu().contiguous()
        tensors[f"{FP8_PREFIX}{name}.scale"] = child._fp8_weight_scale.detach().cpu().contiguous()
    return tensors


def save_fp8_cache(model, cache_dir, source_paths):
    """Write a complete FP8 DiT cache atomically; source files must be stable."""
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        raise FileExistsError(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(source_paths)
    temporary = Path(tempfile.mkdtemp(prefix=f".{cache_dir.name}-", dir=cache_dir.parent))
    try:
        main_state = _state_for_saving(model, exclude_blocks=True)
        save_file(main_state, str(temporary / "main.safetensors"))
        for index, block in enumerate(model.blocks):
            save_file(_state_for_saving(block), str(temporary / f"block-{index:02d}.safetensors"))
        if _fingerprint(source_paths) != fingerprint:
            raise ValueError("source checkpoint changed while writing cache")
        manifest = {"format_version": FORMAT_VERSION, "source": fingerprint,
                    "blocks": len(model.blocks)}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2))
        temporary.rename(cache_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_state(module, path, *, exclude_blocks=False):
    tensors = load_file(str(path), device="cpu")
    fp8_tensors = {name[len(FP8_PREFIX):]: tensors.pop(name)
                   for name in list(tensors) if name.startswith(FP8_PREFIX)}
    expected = {name for name, value in module.state_dict().items()
                if isinstance(value, torch.Tensor) and not (exclude_blocks and name.startswith("blocks."))}
    if set(tensors) != expected:
        raise ValueError(f"cache state mismatch in {path.name}: missing={expected - set(tensors)}, "
                         f"extra={set(tensors) - expected}")
    module.load_state_dict(tensors, strict=False, assign=True)
    for name, child in module.named_modules():
        if not isinstance(child, FP8Linear) or (exclude_blocks and name.startswith("blocks.")):
            continue
        key = f"{name}." if name else ""
        weight = fp8_tensors.pop(key + "weight_nk", None)
        scale = fp8_tensors.pop(key + "scale", None)
        if weight is None or scale is None:
            raise ValueError(f"missing FP8 tensors for {name} in {path.name}")
        child._fp8_weight = weight.t()
        child._fp8_weight_scale = scale
        child._weight_cache_device = torch.device("cpu")
        child._fp16_weight_cpu = None
        child._fp16_bias_cpu = None
    if fp8_tensors:
        raise ValueError(f"unexpected FP8 tensors in {path.name}: {set(fp8_tensors)}")


def load_fp8_cache(model_factory, cache_dir, source_paths):
    """Create the model on meta and hydrate only cached BF16 nonlinears/FP8 linears."""
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported FP8 cache format")
    if manifest.get("source") != _fingerprint(source_paths):
        raise ValueError("source checkpoint changed; rebuild FP8 cache")
    with torch.device("meta"):
        model = model_factory()
    enable_fp8_gemm(model)
    _load_state(model, cache_dir / "main.safetensors", exclude_blocks=True)
    if manifest["blocks"] != len(model.blocks):
        raise ValueError("FP8 cache block count does not match model")
    for index, block in enumerate(model.blocks):
        _load_state(block, cache_dir / f"block-{index:02d}.safetensors")
    if hasattr(model, "init_freqs"):
        model.init_freqs()
    return model
