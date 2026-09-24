# LiveAct Streamed Video Encoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for inline implementation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound host memory used by decoded output to one generation block while preserving the existing MP4 frame sequence.

**Architecture:** Keep the current whole-video export as the default and add an opt-in `--stream_video_output`. A small `video_output.py` helper sends each decoded block to the same imageio FFmpeg writer and uses the same BF16-to-float-to-uint8 conversion as `diffusers.export_to_video`. The existing audio mux remains after writer close.

**Tech Stack:** Python 3.10, PyTorch, imageio FFmpeg, unittest, FFmpeg CLI, RTX 4090.

---

### Task 1: Prove frame parity on a tiny real encode

**Files:** Create `tests/test_video_output.py`; create `video_output.py`.

- [x] Write a unittest that creates two BF16 `[1,3,T,16,16]` blocks, exports their concatenation with `diffusers.utils.export_to_video`, and exports the two blocks through `append_video_block(writer, block)` using one `imageio.get_writer(path, fps=24, quality=5, macro_block_size=16)` context. Compare full RGB24 decoded frame bytes using `ffmpeg -v error -i <path> -f rawvideo -pix_fmt rgb24 -`.
- [x] Run `/home/yg/miniforge3/envs/liveact/bin/python -m unittest discover -s tests -p test_video_output.py -v`; require the expected missing-module failure before implementation.
- [x] Add `append_video_block(writer, block)` to `video_output.py`: for each frame of `((block.cpu().permute(0,2,3,4,1)[0]+1.0)/2)`, call `writer.append_data((frame.float().numpy()*255).astype(np.uint8))`.
- [x] Re-run the test and the full `unittest` suite; require zero failures.

### Task 2: Wire an opt-in path into generate.py

**Files:** Modify `generate.py`; modify `README.md`.

- [x] Add parser flag `--stream_video_output`, default false. In the request loop, open the imageio writer before generation when the flag is true. Replace list accumulation with `append_video_block(writer, _videos)` for this mode. Close the writer through a context manager; keep the existing concat/export branch for default mode. Both branches then call `add_audio_to_video`.
- [x] Run the full `unittest` suite and `python -m py_compile generate.py video_output.py`.
- [x] Document that the flag bounds video frame buffering but does not itself improve generated FPS or make MP4 playable before encoder finalization.

### Task 3: Measure real 4090 parity and memory

**Files:** Modify `docs/benchmarks/2026-09-24-liveact-4090-profile-cache-steps.md`.

- [x] On the same cached 5-second input, compare default and streaming modes with three denoising steps and resident step-0 KV. Verify both full video/audio decodes, frame count, and duration. The synthetic multi-block test establishes exact encoder parity; separate CUDA generation runs need not be pixel-identical.
- [x] Run guarded 30-second streamed and same-version original-export outputs with 3-second process RAM/GPU sampling. Compare sampled peak process RSS and steady-block median; report the sampling limitation and whole-machine interference.
- [x] Commit only after the tests and hardware measurements support the documented result. Do not push or open a PR.
