"""Helpers for LiveAct runtime policy and KV cache allocation."""

import torch


def add_low_memory_arguments(parser):
    parser.add_argument(
        "--disable_compile",
        action="store_true",
        default=False,
        help="Skip torch.compile for the DiT and VAE encoder to reduce first-run time and memory.",
    )
    parser.add_argument(
        "--pageable_block_memory",
        action="store_true",
        default=False,
        help="Use pageable DiT blocks, blocking GPU-to-CPU KV copies, and blockwise video export to reduce host memory use.",
    )
    return parser


def maybe_compile(target, enabled, compiler):
    return compiler(target) if enabled else target


def offload_blocks(blocks, offload_device, pin_cpu_memory, pin_module_memory):
    for block in blocks:
        block.to(offload_device)
        if pin_cpu_memory:
            pin_module_memory(block)


def allocate_kv_caches(token_count, step_count, layer_count, device, fp8,
                       mean_memory, offload, audio_cfg, resident_steps=0,
                       onload_device=None, blocking_cpu_copy=False):
    """Allocate the denoising caches after the model's loading peak has passed."""
    if resident_steps not in (0, 1) or resident_steps > step_count:
        raise ValueError("resident_steps must be 0 or 1 and cannot exceed step_count")
    if resident_steps:
        if not offload or not fp8 or onload_device is None or audio_cfg > 1.0:
            raise ValueError("resident KV requires FP8 CPU offload, a CUDA device, and audio_cfg<=1")
        if torch.device(onload_device).type != "cuda":
            raise ValueError("resident KV requires a CUDA device")
        cache_bytes = token_count * layer_count * (2 * 40 * 128 + 2 * 40 * 4)
        # Release inactive VAE/CLIP allocations before checking device headroom.
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info(onload_device)
        if free_bytes < cache_bytes + 6 * 1024**3:
            raise RuntimeError(
                "insufficient GPU memory for resident KV cache: "
                f"free={free_bytes}, cache={cache_bytes}, reserve={6 * 1024**3} bytes")
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    kv_shape = (1, token_count, 40, 128)
    scale_shape = (1, token_count, 40, 1)

    def make_caches():
        caches = {}
        for step in range(step_count):
            resident = step < resident_steps
            cache_device = onload_device if resident else device
            caches[step] = {
                layer: {
                    "k": torch.zeros(kv_shape, dtype=dtype, device=cache_device),
                    "v": torch.zeros(kv_shape, dtype=dtype, device=cache_device),
                    "k_scale": torch.ones(scale_shape, dtype=torch.float32, device=cache_device) if fp8 else None,
                    "v_scale": torch.ones(scale_shape, dtype=torch.float32, device=cache_device) if fp8 else None,
                    "mean_memory": mean_memory,
                    "offload_cache": offload and not resident,
                    "blocking_cpu_copy": blocking_cpu_copy and offload and not resident,
                    "fp8_kv_cache": fp8,
                }
                for layer in range(layer_count)
            }
        return caches

    return make_caches(), make_caches() if audio_cfg > 1.0 else None
