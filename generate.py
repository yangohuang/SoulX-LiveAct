import os
import numpy as np
import random
import math
import time
import ast
import gc
from tqdm import tqdm
import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
from torchvision import transforms
import torchaudio
import torchaudio.transforms as T

from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE
from util_liveact import *

from wan.modules.clip import CLIPModel
from wan.modules.t5 import T5EncoderModel
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from diffusers.utils import export_to_video

from fp8_gemm import FP8GemmOptions, enable_fp8_gemm
from fp4_gemm import FP4GemmOptions, enable_fp4_gemm
from runtime_options import add_low_memory_arguments, allocate_kv_caches, maybe_compile
from denoising_schedule import denoising_schedule
from fp8_cache import load_fp8_cache, save_fp8_cache
from request_stream import iter_requests, request_key, reset_kv_caches
from prompt_cache import load_prompt_cache, save_prompt_cache
from temporal_continuity import anchor_chunk_start


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt, image and audio"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="480*832",
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--wav2vec_dir",
        type=str,
        default=None,
        help="The path to the wav2vec checkpoint directory.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--offload_cache",
        action="store_true",
        default=False,
        help="Whether to place kv cache on CPU.")
    parser.add_argument(
        "--fp8_kv_cache",
        action="store_true",
        default=False,
        help="Whether to store kv cache in FP8 and dequantize to BF16 on use.")
    parser.add_argument(
        "--fp8_gemm",
        action="store_true",
        default=False,
        help="Whether to enable FP8 GEMM for the model.")
    parser.add_argument(
        "--fp4_gemm",
        action="store_true",
        default=False,
        help="Whether to enable FP4 GEMM for the model.")
    parser.add_argument(
        "--block_offload",
        action="store_true",
        default=False,
        help="Whether to offload WanModel blocks to CPU between block forwards.")
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="The target fps.")
    parser.add_argument(
        "--audio_cfg",
        type=float,
        default=1.0,
        help="Classifier free guidance scale for audio control.")
    parser.add_argument(
        "--dura_print",
        action="store_true",
        default=False,
        help="Whether print duration for every block.")
    parser.add_argument(
        "--input_json",
        type=str,
        default='examples.json',
        help="[meta file] The condition path to generate the video.")
    parser.add_argument(
        "--steam_audio",
        action="store_true",
        default=False,
        help="Whether inference with steaming audio.")
    parser.add_argument(
        "--mean_memory",
        action="store_true",
        default=False,
        help="Whether inference with mean memory strategy.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--denoising_steps",
        type=int,
        choices=(2, 3),
        default=3,
        help="Denoising steps; 3 preserves the original schedule, 2 is an experimental quality/speed tradeoff.")
    parser.add_argument(
        "--fp8_cache_dir",
        type=str,
        default=None,
        help="Load a previously prepared FP8 DiT cache, avoiding BF16 DiT shard loading.")
    parser.add_argument(
        "--build_fp8_cache",
        action="store_true",
        help="Write --fp8_cache_dir after the normal BF16 load and FP8 conversion.")
    parser.add_argument(
        "--serve_stdin",
        action="store_true",
        help="Keep the loaded models resident and accept JSON-line requests on stdin; prompts must appear in --input_json.")
    parser.add_argument(
        "--prompt_cache_dir",
        type=str,
        default=None,
        help="Read or build offline T5 embeddings for the exact prompt catalog in --input_json.")
    parser.add_argument(
        "--resident_kv_steps",
        type=int,
        choices=(0, 1),
        default=0,
        help="Experiment: keep the first denoising step's FP8 KV cache on GPU (about 6.4 GiB at 416x720).")
    parser.add_argument(
        "--motion_anchor_strength",
        type=float,
        default=0.0,
        help="Experiment: blend the next chunk's first two clean latents toward the prior chunk (0 disables it).")

    add_low_memory_arguments(parser)

    args = parser.parse_args()

    return args


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def generate(args):
    startup_started = time.perf_counter()
    if args.fp8_cache_dir and not (args.fp8_gemm and args.block_offload):
        raise ValueError("--fp8_cache_dir requires --fp8_gemm and --block_offload")
    if args.build_fp8_cache and not args.fp8_cache_dir:
        raise ValueError("--build_fp8_cache requires --fp8_cache_dir")
    if args.resident_kv_steps and not (args.offload_cache and args.fp8_kv_cache and args.audio_cfg <= 1.0):
        raise ValueError("--resident_kv_steps requires --offload_cache, --fp8_kv_cache, and audio_cfg<=1")
    if not 0.0 <= args.motion_anchor_strength <= 1.0:
        raise ValueError("--motion_anchor_strength must be between 0 and 1")
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)

    if world_size > 1:
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=world_size,
        )

    if world_size > 1:
        from model_liveact.model_memory_sp import WanModel
    else:
        from model_liveact.model_memory import WanModel

    width, height = [int(_) for _ in args.size.split('*')]
    fps = args.fps
    vae_stride = (4, 8, 8)
    patch_size = (1, 2, 2)
    timestep_values, skip_audio_by_step = denoising_schedule(args.denoising_steps)
    timesteps = [torch.tensor([_]).to(device, dtype=torch.float32) for _ in timestep_values]
    blksz_lst = [6, 8]
    frame_len = (height // (patch_size[1] * vae_stride[1])) * (width // (patch_size[2] * vae_stride[2]))
    kv_cache_tokens = frame_len * sum(blksz_lst) // world_size
    kv_cache_device = 'cpu' if args.offload_cache else device
    kv_cache = None
    kv_cache_null_audio = None

    # Pre-encode all prompts with T5, then free it BEFORE loading the 14B DiT.
    # On a 62G-RAM box, T5 (11G) + DiT (28G) resident together would trigger the OOM killer.
    with open(args.input_json, 'r', encoding='utf-8') as f:
        _pre_input_data = json.load(f)
    prompt_cache_dir = Path(args.prompt_cache_dir) if args.prompt_cache_dir else None
    t5_sources = ([Path(args.ckpt_dir) / "models_t5_umt5-xxl-enc-bf16.pth"] +
                  sorted((Path(args.ckpt_dir) / "google/umt5-xxl").glob("*.json")) +
                  [Path(args.ckpt_dir) / "google/umt5-xxl/spiece.model"]) if prompt_cache_dir else None
    if prompt_cache_dir and prompt_cache_dir.exists():
        cached_ctx = load_prompt_cache(prompt_cache_dir, t5_sources,
                                       [request_key(item) for item in _pre_input_data])
        precomputed_ctx = {
            key: ([value[0][0].to(device, dtype=torch.bfloat16)],
                  {edit_key: tensor.to(device, dtype=torch.bfloat16)
                   for edit_key, tensor in value[1].items()})
            for key, value in cached_ctx.items()
        }
        print(f"Loaded offline T5 prompt cache: {prompt_cache_dir}")
    else:
        text_encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device='cpu' if args.t5_cpu else device,
                                      checkpoint_path=os.path.join(args.ckpt_dir, 'models_t5_umt5-xxl-enc-bf16.pth'),
                                      tokenizer_path=os.path.join(args.ckpt_dir, 'google/umt5-xxl'))
        precomputed_ctx = {}
        for _data in _pre_input_data:
            _key = request_key(_data)
            if _key in precomputed_ctx:
                continue
            _ctx = [text_encoder(texts=_data['prompt'], device='cpu' if args.t5_cpu else device)[0].to(device, dtype=torch.bfloat16)]
            _edit = {
                k: text_encoder(texts=v, device='cpu' if args.t5_cpu else device)[0].to(device, dtype=torch.bfloat16)
                for k, v in _data.get('edit_prompt', {}).items()
            }
            precomputed_ctx[_key] = (_ctx, _edit)
        text_encoder.model = None
        del text_encoder
        if prompt_cache_dir:
            save_prompt_cache(precomputed_ctx, prompt_cache_dir, t5_sources)
            print(f"Saved offline T5 prompt cache: {prompt_cache_dir}")
    torch_gc()
    print(f"LIVEACT_STARTUP t5_s={time.perf_counter() - startup_started:.3f}", flush=True)

    dit_load_started = time.perf_counter()
    cache_dir = Path(args.fp8_cache_dir) if args.fp8_cache_dir else None
    checkpoint_sources = ([Path(args.ckpt_dir) / "config.json"] +
                          sorted(Path(args.ckpt_dir).glob("diffusion_pytorch_model*.safetensors"))) if cache_dir else None
    if cache_dir and cache_dir.exists():
        wan_i2v_model = load_fp8_cache(
            lambda: WanModel.from_config(WanModel.load_config(args.ckpt_dir)),
            cache_dir, checkpoint_sources)
        print(f"Loaded offline FP8 DiT cache: {cache_dir}")
    else:
        if cache_dir and not args.build_fp8_cache:
            raise FileNotFoundError(f"FP8 cache does not exist: {cache_dir}; use --build_fp8_cache once")
        wan_i2v_model = WanModel.from_pretrained(args.ckpt_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        wan_i2v_model = wan_i2v_model.to(dtype=torch.bfloat16)
    print(f"LIVEACT_STARTUP dit_load_s={time.perf_counter() - dit_load_started:.3f}", flush=True)
    for n in range(40):
        wan_i2v_model.blocks[n].self_attn.init_kvidx(frame_len, world_size)

    quant_started = time.perf_counter()
    if args.fp8_gemm and not (cache_dir and cache_dir.exists()):
        enable_fp8_gemm(wan_i2v_model, options=FP8GemmOptions())
        if args.block_offload:
            # Quantize block by block before loading auxiliary models so their
            # BF16 weights do not coexist with the full-precision DiT on CPU.
            # Keep the original lazy path when block offloading is disabled.
            from fp8_gemm import FP8Linear
            _quant_dev = torch.device(f"cuda:{device}")
            for _blk in wan_i2v_model.blocks:
                for _m in _blk.modules():
                    if isinstance(_m, FP8Linear):
                        _m.materialize_fp8_weight(_quant_dev)
                _blk.to('cpu')
            torch_gc()
            print(f"LIVEACT_STARTUP fp8_quant_s={time.perf_counter() - quant_started:.3f}", flush=True)
            if args.build_fp8_cache:
                # The ordinary low-memory path materializes DiT blocks now and
                # leaves small non-block linears lazy. A complete offline cache
                # needs those linears quantized before discarding BF16 shards.
                for _m in wan_i2v_model.modules():
                    if isinstance(_m, FP8Linear) and _m._fp8_weight is None:
                        _m.materialize_fp8_weight(_quant_dev)
                        _m.to('cpu')
                cache_save_started = time.perf_counter()
                save_fp8_cache(wan_i2v_model, cache_dir, checkpoint_sources)
                print(f"LIVEACT_STARTUP cache_save_s={time.perf_counter() - cache_save_started:.3f}", flush=True)
                print(f"Saved offline FP8 DiT cache: {cache_dir}")

    vae = LightVAE(vae_path=os.path.join(args.ckpt_dir, 'Wan2.1_VAE.pth'), dtype=torch.bfloat16, device=device,
                   use_lightvae=False, parallel=(world_size > 1))

    clip = CLIPModel(
        checkpoint_path=os.path.join(args.ckpt_dir, 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'),
        tokenizer_path=os.path.join(args.ckpt_dir, 'xlm-roberta-large'), dtype=torch.bfloat16, device=device)
    clip.model = clip.model.to(device, dtype=torch.bfloat16)

    audio_encoder = Wav2Vec2Model.from_pretrained(
        args.wav2vec_dir, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device, dtype=torch.bfloat16).eval()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_dir, local_files_only=True)

    audio_encoder.feature_extractor._freeze_parameters()
    wan_i2v_model.freqs = wan_i2v_model.freqs.to(device)
    for _model in [wan_i2v_model, clip.model, audio_encoder, vae.model]:
        for name, param in _model.named_parameters():
            param.requires_grad = False

    if args.fp4_gemm:
        print("Enabling FP4 GEMM acceleration...")
        def filter_fn(name, module):
            if "blocks." not in name:
                return False
            return True
        enable_fp4_gemm(wan_i2v_model, options=FP4GemmOptions(), module_filter=filter_fn)
    if args.block_offload:
        for name, child in wan_i2v_model.named_children():
            if name != 'blocks':
                child.to(device)
        wan_i2v_model.enable_block_offload(
            onload_device=torch.device(f"cuda:{device}"),
            pin_cpu_memory=args.pin_block_memory,
        )
    else:
        wan_i2v_model = wan_i2v_model.to(device)
    wan_i2v_model.eval()
    wan_i2v_model = maybe_compile(wan_i2v_model, not args.disable_compile, torch.compile)

    vae.model.eval()
    vae.encode = maybe_compile(vae.encode, not args.disable_compile, torch.compile)

    torch_gc()
    print(f"LIVEACT_STARTUP ready_s={time.perf_counter() - startup_started:.3f}", flush=True)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_rescale_crop_keep_ratio(pil_image, (height, width))),
        transforms.ToTensor(),
        transforms.Resize((height, width)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    if args.serve_stdin:
        print('LIVEACT_READY', flush=True)
    def report_request_error(error):
        print('LIVEACT_ERROR ' + json.dumps({'error': str(error)}, ensure_ascii=False), flush=True)
    requests = iter_requests(_pre_input_data, precomputed_ctx,
                             serve_stdin=args.serve_stdin, stream=sys.stdin,
                             on_error=report_request_error if args.serve_stdin else None)
    for _item_idx, (data, (context, edit_prompts)) in enumerate(requests):
        request_started = time.perf_counter()
        if kv_cache is not None:
            reset_kv_caches(kv_cache)
            reset_kv_caches(kv_cache_null_audio)
        image_path = data['cond_image']
        audio_path = data['cond_audio']
        out_path = data.get('output_path') or (os.path.basename(image_path).split('.')[0] + '_' +
                                               os.path.basename(audio_path).split('.')[0] + '.mp4')

        image = Image.open(image_path).convert("RGB")
        cond_image = transform(image).unsqueeze(1).unsqueeze(0).to(device, torch.bfloat16)  # 1 C 1 H W
        clip.model.to(device)
        clip_context = clip.visual(cond_image)  # 1, 257, 1280
        clip.model.cpu()
        torch_gc()

        audio_ori, sr_ori = torchaudio.load(audio_path)  # y: [channels, time]
        def resample_audio(audio, sr, fps):
            rate = 25 / fps
            effects = [["tempo", f"{rate}"], ]
            y, sr = torchaudio.sox_effects.apply_effects_tensor(audio, sr, effects)
            resampler = T.Resample(sr, 16000)
            return resampler(y) * 3.0, 16000

        audio, sr = resample_audio(audio_ori, sr_ori, fps)
        audio_embedding = get_embedding(audio[0], wav2vec_feature_extractor, audio_encoder, device=device)
        audio_len = audio_ori.size(1) / sr_ori

        ref_target_masks = torch.ones(3, height // vae_stride[1], width // vae_stride[2]).to(device, torch.bfloat16)
        frame_num = (sum(blksz_lst) - 1) * 4 + 1
        msk = get_msk(frame_num, cond_image, vae_stride, device)

        def get_y(frame_num):
            video_frames = torch.zeros(
                1, cond_image.shape[1], frame_num - cond_image.shape[2], height, width
            ).to(cond_image.device, cond_image.dtype)
            padding_frames_pixels_values = torch.concat([cond_image, video_frames], dim=2)
            y = vae.encode(padding_frames_pixels_values.to(vae.device)).to(wan_i2v_model.device).unsqueeze(0)
            y = torch.concat([msk, y], dim=1)
            return y

        y = get_y(frame_num)

        if kv_cache is None:
            kv_cache, kv_cache_null_audio = allocate_kv_caches(
                token_count=kv_cache_tokens,
                step_count=len(timesteps) - 1,
                layer_count=40,
                device=kv_cache_device,
                fp8=args.fp8_kv_cache,
                mean_memory=args.mean_memory,
                offload=args.offload_cache,
                audio_cfg=args.audio_cfg,
                resident_steps=args.resident_kv_steps,
                onload_device=f"cuda:{device}",
            )

        iter_total_num = int(audio_len / (vae_stride[0] * blksz_lst[-1] / fps)) + 1
        print('----iter_total_num=', iter_total_num)
        gen_video_list = []
        torch.manual_seed(args.seed)
        for _ in range(iter_total_num):
            t1 = time.time()
            audio_start_idx, audio_end_idx = 0, frame_num
            if (_ - 1) * blksz_lst[-1] * vae_stride[0] > 0:
                audio_start_idx += (_ - 1) * blksz_lst[-1] * vae_stride[0]
                audio_end_idx += (_ - 1) * blksz_lst[-1] * vae_stride[0]

            if not args.steam_audio:
                audio_embs = get_audio_emb(audio_embedding, audio_start_idx, audio_end_idx, device)
            else:
                audio, sr = resample_audio(
                    audio_ori[:1, int(sr_ori*(audio_start_idx/fps)):int(sr_ori*((audio_end_idx+2)/fps))], sr_ori, fps
                )
                audio_embedding = get_embedding(audio[0], wav2vec_feature_extractor, audio_encoder, device=device)
                audio_embs = get_audio_emb(audio_embedding, 0, frame_num, device)

            y_cut = y[:, :, :frame_num // 4 + 1, ...]

            _context = context
            if edit_prompts:
                for k, v in edit_prompts.items():
                    if ast.literal_eval(k)[0] <= _ <= ast.literal_eval(k)[1]:
                        _context = [v]
                        break

            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                f = _ if _ <= 1 else 1
                latent = torch.randn(16, blksz_lst[f], height // vae_stride[1], width // vae_stride[2],
                                     dtype=torch.bfloat16, device=device)
                for i in tqdm(range(len(timesteps) - 1)):
                    timestep = timesteps[i]
                    arg_c = {'context': _context, 'clip_fea': clip_context, 'ref_target_masks': ref_target_masks,
                             'audio': audio_embs, 'y': y_cut[:, :, sum(blksz_lst[:f]):sum(blksz_lst[:f + 1])],
                             'start_idx': sum(blksz_lst[:f]) * frame_len, 'end_idx': sum(blksz_lst[:f + 1]) * frame_len,
                             'update_cache': _ > 1}
                    noise_pred = wan_i2v_model([latent.to(device)], t=timestep, kv_cache=kv_cache[i],
                                               skip_audio=skip_audio_by_step[i], **arg_c)[0]

                    if args.audio_cfg>1.0 and not skip_audio_by_step[i]:
                        arg_null_audio = \
                            {'context': _context, 'clip_fea': clip_context, 'ref_target_masks': ref_target_masks,
                             'audio': torch.zeros_like(audio_embs), 'y': y_cut[:, :, sum(blksz_lst[:f]):sum(blksz_lst[:f + 1])],
                             'start_idx': sum(blksz_lst[:f]) * frame_len, 'end_idx': sum(blksz_lst[:f + 1]) * frame_len,
                             'update_cache': _ > 1}
                        noise_pred_drop_audio = wan_i2v_model([latent.to(device)], t=timestep, kv_cache=kv_cache_null_audio[i],
                                                              **arg_null_audio)[0]
                        noise_pred = noise_pred_drop_audio + args.audio_cfg * (noise_pred - noise_pred_drop_audio)

                    dt = timesteps[i] - timesteps[i + 1]
                    dt = dt / 1000
                    # latent = latent + (-noise_pred) * dt[0]
                    x0_pred = latent + (-noise_pred) * (timesteps[i][0]/1000 - 0.0)
                    if f > 0 and args.motion_anchor_strength > 0:
                        x0_pred = anchor_chunk_start(x0_pred, pre_latent, args.motion_anchor_strength)
                    latent = (1-timesteps[i+1][0]/1000)*x0_pred + torch.randn_like(x0_pred)*(timesteps[i+1][0]/1000)

                if f == 0:
                    _latent = latent
                    _videos = vae.decode(_latent.squeeze(0))
                else:
                    _latent = torch.concat([pre_latent[:, -3:], latent], dim=1)
                    _videos = vae.decode(_latent.squeeze(0))[:, :, 9:]
                pre_latent = latent
                gen_video_list.append(_videos.cpu())

                if args.dura_print:
                    torch.cuda.synchronize()
                    if rank == 0:
                        t2 = time.time()
                        dura = blksz_lst[f] * vae_stride[0] / fps * 1000
                        print(f"Done Block {_}: duration {dura}ms video cost {(t2 - t1) * 1000:.2f} ms")
        torch.cuda.synchronize()
        # torch_gc()

        videos = (torch.concat(gen_video_list, dim=2).permute((0, 2, 3, 4, 1))[0] + 1.0) / 2
        video_path = 'tmp.mp4'
        export_to_video(videos[:, ...].float().cpu().numpy(), video_path, fps=fps)
        add_audio_to_video(video_path, audio_path, out_path)
        if args.serve_stdin:
            print('LIVEACT_RESULT ' + json.dumps({
                'output_path': out_path,
                'elapsed_s': round(time.perf_counter() - request_started, 3),
            }, ensure_ascii=False), flush=True)
        del videos, gen_video_list
        gc.collect()

        torch.cuda.synchronize()
        # torch_gc()

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    args = _parse_args()
    generate(args)
