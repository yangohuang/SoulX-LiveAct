"""Fingerprint LiveAct conditioning and generation boundaries across processes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


def fingerprint(tensor: torch.Tensor) -> dict:
    value = tensor.detach().contiguous().cpu()
    data = value.view(torch.uint8).numpy().tobytes()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, required=True)
    cudnn = parser.add_mutually_exclusive_group()
    cudnn.add_argument("--deterministic-cudnn", action="store_true",
                       help="Disable cuDNN benchmarking and require deterministic cuDNN kernels.")
    cudnn.add_argument("--benchmark-off-only", action="store_true",
                       help="Disable cuDNN benchmarking without requiring deterministic kernels.")
    cudnn.add_argument("--deterministic-only", action="store_true",
                       help="Require deterministic cuDNN kernels but keep benchmarking enabled.")
    args = parser.parse_args()
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    records = []
    tensor_dir = prefix.with_name(prefix.name + "-tensors")
    tensor_dir.mkdir(exist_ok=True)

    def record(name: str, tensor: torch.Tensor, *, save: bool = False) -> None:
        entry = {"name": name, **fingerprint(tensor)}
        records.append(entry)
        if save:
            torch.save(tensor.detach().cpu(), tensor_dir / f"{len(records):03d}-{name}.pt")

    import generate
    from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE
    from model_liveact.model_memory import WanModel

    if args.deterministic_cudnn or args.benchmark_off_only:
        torch.backends.cudnn.benchmark = False
    if args.deterministic_cudnn or args.deterministic_only:
        torch.backends.cudnn.deterministic = True

    visual = generate.CLIPModel.visual

    def traced_visual(self, videos):
        record("reference_image", videos)
        out = visual(self, videos)
        record("clip_context", out, save=True)
        return out

    generate.CLIPModel.visual = traced_visual
    get_embedding = generate.get_embedding

    def traced_embedding(*args, **kwargs):
        record("resampled_audio", args[0])
        out = get_embedding(*args, **kwargs)
        record("wav2vec_embedding", out, save=True)
        return out

    generate.get_embedding = traced_embedding
    encode = LightVAE.encode
    decode = LightVAE.decode

    def traced_encode(self, video, *positional, **kwargs):
        record("vae_encode_input", video)
        out = encode(self, video, *positional, **kwargs)
        record("vae_reference_latent", out, save=True)
        return out

    def traced_decode(self, z, *positional, **kwargs):
        index = sum(row["name"].startswith("vae_decode_input_") for row in records)
        record(f"vae_decode_input_{index}", z, save=True)
        out = decode(self, z, *positional, **kwargs)
        record(f"vae_decode_output_{index}", out, save=True)
        return out

    LightVAE.encode = traced_encode
    LightVAE.decode = traced_decode

    forward = WanModel.forward
    calls = 0

    def traced_forward(self, *positional, **kwargs):
        nonlocal calls
        index = calls
        calls += 1
        record(f"dit_input_{index}", positional[0][0], save=index == 0)
        if index == 0:
            for key in ("clip_fea", "audio", "y"):
                record(f"dit_{key}_0", kwargs[key], save=True)
            record("dit_context_0", kwargs["context"][0], save=True)
        out = forward(self, *positional, **kwargs)
        record(f"dit_output_{index}", out[0], save=index == 0)
        return out

    WanModel.forward = traced_forward
    original_manual_seed = torch.manual_seed
    original_randn = torch.randn
    original_randn_like = torch.randn_like
    after_seed = False
    noise_count = 0

    def traced_manual_seed(seed):
        nonlocal after_seed
        after_seed = True
        records.append({"name": "manual_seed", "value": int(seed)})
        return original_manual_seed(seed)

    def traced_randn(*positional, **kwargs):
        nonlocal noise_count
        out = original_randn(*positional, **kwargs)
        if after_seed:
            record(f"randn_{noise_count}", out, save=noise_count == 0)
            noise_count += 1
        return out

    def traced_randn_like(*positional, **kwargs):
        nonlocal noise_count
        out = original_randn_like(*positional, **kwargs)
        if after_seed:
            record(f"randn_like_{noise_count}", out)
            noise_count += 1
        return out

    torch.manual_seed = traced_manual_seed
    torch.randn = traced_randn
    torch.randn_like = traced_randn_like

    request = [{"prompt": "一个人在说话", "cond_image": "examples/image/1.png",
                "cond_audio": "/tmp/liveact-profile-1p5s.wav",
                "output_path": str(prefix.with_suffix(".mp4"))}]
    request_path = prefix.with_name(prefix.name + "-request.json")
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    argv = ["generate.py", "--size", "224*384", "--ckpt_dir", "checkpoints/LiveAct",
            "--wav2vec_dir", "checkpoints/chinese-wav2vec2-base", "--fps", "24",
            "--seed", "42", "--denoising_steps", "3", "--input_json", str(request_path),
            "--fp8_gemm", "--fp8_kv_cache", "--offload_cache", "--block_offload",
            "--t5_cpu", "--disable_compile", "--resident_kv_steps", "1",
            "--fp8_cache_dir", "/tmp/liveact-fp8-cache",
            "--prompt_cache_dir", "/tmp/liveact-prompt-cache"]
    sys.argv = argv
    try:
        generate.generate(generate._parse_args())
    finally:
        prefix.with_name(prefix.name + "-fingerprints.json").write_text(
            json.dumps({"argv": argv, "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "cudnn_deterministic": torch.backends.cudnn.deterministic,
                        "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
