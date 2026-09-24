"""Small, offline cache of T5 contexts for a fixed LiveAct prompt catalog."""

import json
import shutil
import tempfile
from pathlib import Path

from safetensors.torch import load_file, save_file


def _fingerprint(source_paths):
    return [{"path": str(Path(path).resolve()), "size": Path(path).stat().st_size,
             "mtime_ns": Path(path).stat().st_mtime_ns} for path in source_paths]


def save_prompt_cache(contexts, cache_dir, source_paths):
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        raise FileExistsError(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    source = _fingerprint(source_paths)
    temporary = Path(tempfile.mkdtemp(prefix=f".{cache_dir.name}-", dir=cache_dir.parent))
    try:
        tensors = {}
        catalog = {}
        for index, (request_key, (context, edits)) in enumerate(contexts.items()):
            context_name = f"context-{index}"
            tensors[context_name] = context[0].detach().cpu().contiguous()
            edit_names = {}
            for edit_index, (edit_key, value) in enumerate(edits.items()):
                name = f"edit-{index}-{edit_index}"
                tensors[name] = value.detach().cpu().contiguous()
                edit_names[edit_key] = name
            catalog[request_key] = {"context": context_name, "edits": edit_names}
        save_file(tensors, str(temporary / "contexts.safetensors"))
        if _fingerprint(source_paths) != source:
            raise ValueError("T5 source changed while writing prompt cache")
        (temporary / "manifest.json").write_text(json.dumps({
            "format_version": 1, "source": source, "catalog": catalog,
        }, ensure_ascii=False, indent=2))
        temporary.rename(cache_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_prompt_cache(cache_dir, source_paths, expected_keys):
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported prompt cache format")
    if manifest["source"] != _fingerprint(source_paths):
        raise ValueError("T5 source changed; rebuild prompt cache")
    catalog = manifest["catalog"]
    if set(catalog) != set(expected_keys):
        raise ValueError("prompt catalog changed; rebuild prompt cache")
    tensors = load_file(str(cache_dir / "contexts.safetensors"), device="cpu")
    return {key: ([tensors[entry["context"]]],
                  {edit_key: tensors[name] for edit_key, name in entry["edits"].items()})
            for key, entry in catalog.items()}
