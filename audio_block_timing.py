"""Read-only index/availability audit for LiveAct's 24-FPS audio-driven blocks."""

import argparse
import json
from pathlib import Path


# Frozen from generate.py: blksz_lst=[6,8], frame_num=(sum(blksz_lst)-1)*4+1;
# util_liveact.get_audio_emb selects each center frame and its ±2 neighbors.
FPS = 24
FIRST_BLOCK_FRAMES = (6 - 1) * 4 + 1
LATER_BLOCK_FRAMES = 8 * 4
AUDIO_WINDOW_FRAMES = (6 + 8 - 1) * 4 + 1
AUDIO_RADIUS_FRAMES = 2


def block_timing(index: int, audio_frames: int) -> dict:
    """Describe selected audio indices and output frames for one existing block."""
    if index < 0 or audio_frames < 1:
        raise ValueError("block index must be nonnegative and audio nonempty")
    audio_start = 0 if index <= 1 else (index - 1) * LATER_BLOCK_FRAMES
    audio_end = audio_start + AUDIO_WINDOW_FRAMES  # exclusive centers
    output_first = 0 if index == 0 else FIRST_BLOCK_FRAMES + (index - 1) * LATER_BLOCK_FRAMES
    output_frames = FIRST_BLOCK_FRAMES if index == 0 else LATER_BLOCK_FRAMES
    first_selected = audio_start - AUDIO_RADIUS_FRAMES
    last_selected = audio_end - 1 + AUDIO_RADIUS_FRAMES
    clamped = sum(not (0 <= center + offset < audio_frames)
                  for center in range(audio_start, audio_end)
                  for offset in range(-AUDIO_RADIUS_FRAMES, AUDIO_RADIUS_FRAMES + 1))
    # --steam_audio slices through (audio_end_idx+2)/fps, exclusive in samples.
    slice_end = audio_end + AUDIO_RADIUS_FRAMES
    return {
        "block_index": index,
        "output_first_frame": output_first,
        "output_last_frame": output_first + output_frames - 1,
        "audio_start_idx": audio_start,
        "audio_end_idx_exclusive": audio_end,
        "audio_first_unclamped": first_selected,
        "audio_last_unclamped": last_selected,
        "audio_first_clamped": max(0, first_selected),
        "audio_last_clamped": min(audio_frames - 1, last_selected),
        "audio_window_positions": AUDIO_WINDOW_FRAMES * (2 * AUDIO_RADIUS_FRAMES + 1),
        "clamped_window_positions": clamped,
        "streaming_audio_slice_end_exclusive": slice_end,
        "audio_slice_lead_from_first_output_seconds": (slice_end - output_first) / FPS,
        "audio_slice_lead_from_last_output_seconds":
            (slice_end - (output_first + output_frames - 1)) / FPS,
    }


def period_phases(period_frames: int, count: int) -> list[int]:
    if period_frames < 1 or count < 1:
        raise ValueError("period and count must be positive")
    return [(index * period_frames) % LATER_BLOCK_FRAMES for index in range(count)]


def scenario(period_seconds: int, repeat_count: int) -> dict:
    if period_seconds < 1 or repeat_count < 1:
        raise ValueError("period duration and repeat count must be positive")
    period_frames = period_seconds * FPS
    audio_frames = period_frames * repeat_count
    # Same int(audio_len / (4*8/fps)) + 1 as generate.py for integer-frame audio.
    block_count = audio_frames // LATER_BLOCK_FRAMES + 1
    blocks = [block_timing(index, audio_frames) for index in range(block_count)]
    boundaries = []
    for index in range(1, repeat_count):
        frame = index * period_frames
        crossing = [block for block in blocks
                    if block["audio_first_unclamped"] < frame <= block["audio_last_unclamped"]]
        edge_margin = 2 * FPS
        outside_edge_margin = sorted({output_frame for block in crossing
                                      for output_frame in range(block["output_first_frame"],
                                                                block["output_last_frame"] + 1)
                                      if 0 <= output_frame < audio_frames
                                      and not (frame - edge_margin <= output_frame <
                                               frame + edge_margin)})
        boundaries.append({
            "period_boundary_frame": frame,
            "phase_mod_32_output_frames": frame % LATER_BLOCK_FRAMES,
            "audio_window_crossing_block_indices": [block["block_index"]
                                                    for block in crossing],
            "crossing_block_output_frames_outside_2s_margin": outside_edge_margin,
        })
    return {
        "period_seconds": period_seconds,
        "repeat_count": repeat_count,
        "fps": FPS,
        "first_block_output_frames": FIRST_BLOCK_FRAMES,
        "later_block_output_frames": LATER_BLOCK_FRAMES,
        "audio_window_center_frames": AUDIO_WINDOW_FRAMES,
        "audio_context_radius_frames": AUDIO_RADIUS_FRAMES,
        "period_frames": period_frames,
        "audio_frames": audio_frames,
        "block_count": block_count,
        "period_start_phase_mod_32": period_phases(period_frames, repeat_count),
        "period_boundaries": boundaries,
        "predicted_pre_encoder_frames": blocks[-1]["output_last_frame"] + 1,
        "blocks": blocks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period-seconds", type=int, required=True)
    parser.add_argument("--repeat-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = scenario(args.period_seconds, args.repeat_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "blocks"}, indent=2))


if __name__ == "__main__":
    main()
